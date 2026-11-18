#include "test_utils.h"

#include <cstdint>
#include <errno.h>

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t alloc_size = 1 << 20;
    const size_t seg_size = 4096;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    uGDSIoSegment_t segs[4];

    /* 1. Null / zero-count / oversize argument rejection. */
    {
        uGDSIoSegment_t one = { d_buf, 0, seg_size };
        ssize_t ret = uGDSReadv(fh, nullptr, 1, 0);
        if (ret != -EINVAL) TEST_FAIL("null segs: %zd", ret);
        ret = uGDSReadv(fh, &one, 0, 0);
        if (ret != -EINVAL) TEST_FAIL("zero nr_segs: %zd", ret);
        ret = uGDSReadv(fh, &one, UGDS_IOV_MAX + 1, 0);
        if (ret != -EINVAL) TEST_FAIL("oversize nr_segs: %zd", ret);
    }

    /* 2. Unregistered base pointer rejection (surfaced as -EINVAL). */
    {
        uGDSIoSegment_t unreg = { (char*)d_buf + alloc_size, 0, seg_size };
        ssize_t ret = uGDSWritev(fh, &unreg, 1, 0);
        if (ret != -EINVAL) TEST_FAIL("unregistered base: %zd", ret);
    }

    /* 3. Misaligned segment offset rejection. */
    {
        uGDSIoSegment_t mis = { d_buf, 1, seg_size };
        ssize_t ret = uGDSReadv(fh, &mis, 1, 0);
        if (ret != -EINVAL) TEST_FAIL("misaligned offset: %zd", ret);
    }

    /* 4. Segment size not block-multiple rejection.  Use a size that is
     * unaligned under any LBA format (512B or 4K); a seg_size-512 style
     * tweak only misaligns on 4K-LBA devices. */
    {
        uGDSIoSegment_t odd = { d_buf, 0, seg_size - 1 };
        ssize_t ret = uGDSReadv(fh, &odd, 1, 0);
        if (ret != -EINVAL) TEST_FAIL("odd size: %zd", ret);
    }

    /* 5. Segment past the end of the registration rejection. */
    {
        uGDSIoSegment_t past = { d_buf, alloc_size, seg_size };
        ssize_t ret = uGDSReadv(fh, &past, 1, 0);
        if (ret != -EINVAL) TEST_FAIL("past-end offset: %zd", ret);
    }

    /* 6. Batch vectored preflight: an entry whose values are well-formed
     * and aligned but outside the registered buffer range fails as a
     * terminal FAILED event with -EINVAL, not a submit error.  (A
     * misaligned segment offset is a value error and is rejected at
     * submit, matching the plain uGDSBatchIOSubmit contract.) */
    {
        uGDSBatchHandle_t batch = nullptr;
        st = uGDSBatchIOSetUp(&batch, fh, 4);
        ASSERT_OK(st, "BatchIOSetUp");

        uGDSIoSegment_t bad_seg = { d_buf, alloc_size, seg_size };  /* past end */
        uGDSIOSegParams_t iocb;
        memset(&iocb, 0, sizeof(iocb));
        iocb.segs = &bad_seg;
        iocb.nr_segs = 1;
        iocb.file_offset = 0;
        iocb.opcode = UGDS_READ;
        iocb.cookie = (void*)0xBAD1;

        st = uGDSBatchIOSubmitv(batch, 1, &iocb, 0);
        ASSERT_OK(st, "BatchIOSubmitv accepts bad entry");

        unsigned nr_out = 0;
        uGDSIOEvents_t ev[4];
        st = uGDSBatchIOGetStatus(batch, 1, &nr_out, ev, nullptr);
        ASSERT_OK(st, "BatchIOGetStatus");
        if (nr_out != 1) TEST_FAIL("expected 1 event, got %u", nr_out);
        if (ev[0].status != UGDS_BATCH_FAILED)
            TEST_FAIL("expected FAILED(0x%x), got 0x%x",
                      UGDS_BATCH_FAILED, ev[0].status);
        if (ev[0].ret != -EINVAL)
            TEST_FAIL("expected -EINVAL, got %zd", ev[0].ret);

        uGDSBatchIODestroy(batch);
    }

    /* 7. Async vectored with an unregistered base must refuse at
     * enqueue time, before any stream work is queued. */
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        uGDSIoSegment_t unreg[2];
        unreg[0].base = d_buf;
        unreg[0].offset = 0;
        unreg[0].size = seg_size;
        unreg[1].base = (char*)d_buf + alloc_size;
        unreg[1].offset = 0;
        unreg[1].size = seg_size;

        off_t off = 0;
        ssize_t n = -1;
        uGDSError_t rst = uGDSReadvAsync(fh, unreg, 2, &off, &n,
                                         (void*)stream);
        if (rst.err == UGDS_SUCCESS)
            TEST_FAIL("ReadvAsync accepted unregistered base");
        cudaStreamDestroy(stream);
    }

    (void)segs;
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
