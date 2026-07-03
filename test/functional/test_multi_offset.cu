#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t chunk = 4096;
    const int n_chunks = 4;
    const size_t alloc_size = 65536;
    const uint32_t patterns[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    size_t n_words = chunk / sizeof(uint32_t);

    // Write 4 different patterns at 4 offsets
    for (int i = 0; i < n_chunks; i++) {
        fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, patterns[i], n_words);
        cudaDeviceSynchronize();

        off_t file_off = (off_t)(i * chunk);
        ssize_t ret = uGDSWrite(fh, d_buf, chunk, file_off, 0);
        if (ret < 0) TEST_FAIL("uGDSWrite chunk %d returned %zd", i, ret);
        if ((size_t)ret != chunk) TEST_FAIL("uGDSWrite chunk %d short: %zd / %zu", i, ret, chunk);
    }

    // Read each back and verify
    uint32_t* h_buf = (uint32_t*)malloc(chunk);
    for (int i = 0; i < n_chunks; i++) {
        cudaMemset(d_buf, 0, alloc_size);
        cudaDeviceSynchronize();

        off_t file_off = (off_t)(i * chunk);
        ssize_t ret = uGDSRead(fh, d_buf, chunk, file_off, 0);
        if (ret < 0) {
            free(h_buf);
            TEST_FAIL("uGDSRead chunk %d returned %zd", i, ret);
        }
        if ((size_t)ret != chunk) {
            free(h_buf);
            TEST_FAIL("uGDSRead chunk %d short: %zd / %zu", i, ret, chunk);
        }

        cudaMemcpy(h_buf, d_buf, chunk, cudaMemcpyDeviceToHost);
        for (size_t w = 0; w < n_words; w++) {
            if (h_buf[w] != patterns[i]) {
                free(h_buf);
                TEST_FAIL("chunk %d word %zu: 0x%08X != 0x%08X", i, w, h_buf[w], patterns[i]);
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
