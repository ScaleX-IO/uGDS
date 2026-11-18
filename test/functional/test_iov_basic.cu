#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t alloc_size = 1 << 20;   /* 1 MiB */
    const size_t seg_size = 4096;
    const unsigned nr_segs = 16;
    const uint32_t pattern = 0x51A7E000;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    /* Fill the whole buffer with the pattern, then write segments
     * striding through it as a scattered list. */
    size_t n_words = alloc_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
        (uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    uGDSIoSegment_t segs[nr_segs];
    unsigned k = 0;
    for (unsigned i = 0; i < nr_segs; i++) {
        /* Interleave two regions so the list is genuinely scattered:
         * even segments from the first half, odd from the second. */
        size_t region = (i % 2) ? (alloc_size / 2) : 0;
        size_t slot = (i / 2) * 2 * seg_size;
        segs[k].base = d_buf;
        segs[k].offset = region + slot;
        segs[k].size = seg_size;
        k++;
    }

    ssize_t ret = uGDSWritev(fh, segs, nr_segs, 0);
    if (ret < 0) TEST_FAIL("uGDSWritev returned %zd", ret);
    if ((size_t)ret != nr_segs * seg_size)
        TEST_FAIL("uGDSWritev short: %zd / %zu", ret, nr_segs * seg_size);

    /* Clear only the transferred segments; leave gaps untouched. */
    for (unsigned i = 0; i < nr_segs; i++)
        cudaMemset((char*)d_buf + segs[i].offset, 0, seg_size);
    cudaDeviceSynchronize();

    ret = uGDSReadv(fh, segs, nr_segs, 0);
    if (ret < 0) TEST_FAIL("uGDSReadv returned %zd", ret);
    if ((size_t)ret != nr_segs * seg_size)
        TEST_FAIL("uGDSReadv short: %zd / %zu", ret, nr_segs * seg_size);

    uint32_t* h_buf = (uint32_t*)malloc(alloc_size);
    cudaMemcpy(h_buf, d_buf, alloc_size, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < nr_segs; i++) {
        uint32_t* seg = (uint32_t*)((char*)h_buf + segs[i].offset);
        for (size_t w = 0; w < seg_size / sizeof(uint32_t); w++) {
            if (seg[w] != pattern) {
                free(h_buf);
                TEST_FAIL("segment %u word %zu: 0x%08X != 0x%08X",
                          i, w, seg[w], pattern);
            }
        }
    }
    free(h_buf);

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
