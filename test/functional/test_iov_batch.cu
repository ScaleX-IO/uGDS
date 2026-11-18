#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t alloc_size = 1 << 20;
    const size_t seg_size = 4096;
    const unsigned nr_entries = 8;
    const unsigned segs_per_entry = 4;
    const uint32_t pattern = 0xBA7C5E6A;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    /* Seed the device with a known pattern via the sync write path. */
    size_t n_words = alloc_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
        (uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    ssize_t ret = uGDSWrite(fh, d_buf, alloc_size, 0, 0);
    if (ret < 0) TEST_FAIL("seed write returned %zd", ret);
    if ((size_t)ret != alloc_size) TEST_FAIL("seed write short: %zd", ret);

    cudaMemset(d_buf, 0, alloc_size);
    cudaDeviceSynchronize();

    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, nr_entries);
    ASSERT_OK(st, "BatchIOSetUp");

    /* Each vectored entry reads several scattered segments from a
     * distinct file range; plain and vectored submits may be mixed,
     * so keep this batch vectored-only for a clean check. */
    uGDSIOSegParams_t iocb[nr_entries];
    for (unsigned e = 0; e < nr_entries; e++) {
        static uGDSIoSegment_t entry_segs[nr_entries][segs_per_entry];
        for (unsigned s = 0; s < segs_per_entry; s++) {
            entry_segs[e][s].base = d_buf;
            entry_segs[e][s].offset = ((e * segs_per_entry + s) % 64) * seg_size;
            entry_segs[e][s].size = seg_size;
        }
        memset(&iocb[e], 0, sizeof(iocb[e]));
        iocb[e].segs = entry_segs[e];
        iocb[e].nr_segs = segs_per_entry;
        iocb[e].file_offset = (off_t)(e * segs_per_entry) * seg_size;
        iocb[e].opcode = UGDS_READ;
        iocb[e].cookie = (void*)(uintptr_t)(0xE000 + e);
    }

    st = uGDSBatchIOSubmitv(batch, nr_entries, iocb, 0);
    ASSERT_OK(st, "BatchIOSubmitv");

    unsigned total = 0;
    uGDSIOEvents_t events[nr_entries];
    while (total < nr_entries) {
        unsigned nr_out = 0;
        st = uGDSBatchIOGetStatus(batch, nr_entries - total, &nr_out,
                                  events, nullptr);
        ASSERT_OK(st, "BatchIOGetStatus");
        if (nr_out == 0) TEST_FAIL("GetStatus starved at %u/%u", total, nr_entries);
        for (unsigned i = 0; i < nr_out; i++) {
            if (events[i].status != UGDS_BATCH_COMPLETE)
                TEST_FAIL("entry status: expected COMPLETE, got 0x%x",
                          events[i].status);
            if (events[i].ret != (ssize_t)(segs_per_entry * seg_size))
                TEST_FAIL("entry ret: expected %zu, got %zd",
                          segs_per_entry * seg_size, events[i].ret);
        }
        total += nr_out;
    }

    /* Verify every scattered segment landed at the right place. */
    uint32_t* h_buf = (uint32_t*)malloc(alloc_size);
    cudaMemcpy(h_buf, d_buf, alloc_size, cudaMemcpyDeviceToHost);
    for (unsigned e = 0; e < nr_entries; e++) {
        for (unsigned s = 0; s < segs_per_entry; s++) {
            size_t off = ((e * segs_per_entry + s) % 64) * seg_size;
            uint32_t* seg = (uint32_t*)((char*)h_buf + off);
            for (size_t w = 0; w < seg_size / sizeof(uint32_t); w++) {
                if (seg[w] != pattern) {
                    free(h_buf);
                    TEST_FAIL("entry %u seg %u word %zu mismatch", e, s, w);
                }
            }
        }
    }
    free(h_buf);

    uGDSBatchIODestroy(batch);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
