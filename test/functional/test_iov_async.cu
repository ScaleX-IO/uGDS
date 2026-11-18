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
    const unsigned nr_segs = 8;
    const uint32_t pattern = 0xA0D1E5C9;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    /* Seed the file with the pattern through the sync path. */
    size_t n_words = alloc_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
        (uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    ssize_t ret = uGDSWrite(fh, d_buf, alloc_size, 0, 0);
    if (ret < 0) TEST_FAIL("seed write returned %zd", ret);
    if ((size_t)ret != alloc_size) TEST_FAIL("seed write short: %zd", ret);

    cudaMemset(d_buf, 0, alloc_size);
    cudaDeviceSynchronize();

    uGDSIoSegment_t segs[nr_segs];
    for (unsigned i = 0; i < nr_segs; i++) {
        segs[i].base = d_buf;
        segs[i].offset = (i * 2 * seg_size) % (alloc_size / 2);
        segs[i].size = seg_size;
    }

    /* Late-binding outputs live on the host, like the scalar async API. */
    off_t file_offset = 0;
    ssize_t bytes_read = -1;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    st = uGDSReadvAsync(fh, segs, nr_segs, &file_offset, &bytes_read,
                        (void*)stream);
    ASSERT_OK(st, "ReadvAsync enqueue");

    /* The segs array must stay valid until the callback runs; the sync
     * here stands in for the kernel that consumes the data. */
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    if (bytes_read != (ssize_t)(nr_segs * seg_size))
        TEST_FAIL("bytes_read: expected %zu, got %zd",
                  nr_segs * seg_size, bytes_read);

    /* Same buffer, write back through the async vectored path. */
    ssize_t bytes_written = -1;
    st = uGDSWritevAsync(fh, segs, nr_segs, &file_offset, &bytes_written,
                         (void*)stream);
    ASSERT_OK(st, "WritevAsync enqueue");
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    if (bytes_written != (ssize_t)(nr_segs * seg_size))
        TEST_FAIL("bytes_written: expected %zu, got %zd",
                  nr_segs * seg_size, bytes_written);

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

    cudaStreamDestroy(stream);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
