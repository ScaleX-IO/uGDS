#include "test_utils.h"
#include <thread>
#include <atomic>

static std::atomic<int> g_errors{0};

static void worker(int id, uGDSHandle_t fh) {
    const size_t alloc_size = 65536;
    const size_t io_size = 4096;
    const uint32_t pattern = 0xBEEF0000 | (uint32_t)id;
    const off_t file_off = (off_t)(id + 1) * alloc_size;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) {
        fprintf(stderr, "worker %d: cudaMalloc failed\n", id);
        g_errors++;
        return;
    }

    uGDSError_t st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    if (st.err != UGDS_SUCCESS) {
        fprintf(stderr, "worker %d: BufRegister failed: %s\n", id, uGDS_status_error(st.err));
        cudaFree(d_buf);
        g_errors++;
        return;
    }

    size_t n_words = io_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    ssize_t ret = uGDSWrite(fh, d_buf, io_size, file_off, 0);
    if (ret < 0 || (size_t)ret != io_size) {
        fprintf(stderr, "worker %d: uGDSWrite failed: %zd\n", id, ret);
        uGDSBufDeregister(d_buf);
        cudaFree(d_buf);
        g_errors++;
        return;
    }

    cudaMemset(d_buf, 0, alloc_size);
    cudaDeviceSynchronize();
    ret = uGDSRead(fh, d_buf, io_size, file_off, 0);
    if (ret < 0 || (size_t)ret != io_size) {
        fprintf(stderr, "worker %d: uGDSRead failed: %zd\n", id, ret);
        uGDSBufDeregister(d_buf);
        cudaFree(d_buf);
        g_errors++;
        return;
    }

    uint32_t* h_buf = (uint32_t*)malloc(io_size);
    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != pattern) {
            fprintf(stderr, "worker %d: mismatch at word %zu: 0x%08X != 0x%08X\n",
                    id, i, h_buf[i], pattern);
            free(h_buf);
            uGDSBufDeregister(d_buf);
            cudaFree(d_buf);
            g_errors++;
            return;
        }
    }
    free(h_buf);

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const int n_threads = 4;
    std::thread threads[n_threads];

    for (int i = 0; i < n_threads; i++)
        threads[i] = std::thread(worker, i, fh);

    for (int i = 0; i < n_threads; i++)
        threads[i].join();

    close_handle(fh);
    uGDSDriverClose();

    if (g_errors.load() != 0)
        TEST_FAIL("%d worker(s) reported errors", g_errors.load());

    TEST_PASS();
}
