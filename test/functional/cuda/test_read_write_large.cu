#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t io_size = 1 * 1024 * 1024;  // 1 MB
    const uint32_t pattern = 0xA5A5A5A5;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, io_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, io_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    // Fill with pattern
    size_t n_words = io_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    // Write 1MB at offset 0
    ssize_t ret = uGDSWrite(fh, d_buf, io_size, 0, 0);
    if (ret < 0) TEST_FAIL("uGDSWrite returned %zd", ret);
    if ((size_t)ret != io_size) TEST_FAIL("uGDSWrite short: %zd / %zu", ret, io_size);

    // Clear GPU buffer
    cudaMemset(d_buf, 0, io_size);
    cudaDeviceSynchronize();

    // Read back
    ret = uGDSRead(fh, d_buf, io_size, 0, 0);
    if (ret < 0) TEST_FAIL("uGDSRead returned %zd", ret);
    if ((size_t)ret != io_size) TEST_FAIL("uGDSRead short: %zd / %zu", ret, io_size);

    // Verify on host
    uint32_t* h_buf = (uint32_t*)malloc(io_size);
    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != pattern) {
            free(h_buf);
            TEST_FAIL("mismatch at word %zu: 0x%08X != 0x%08X", i, h_buf[i], pattern);
        }
    }
    free(h_buf);

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
