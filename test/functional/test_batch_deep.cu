#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const unsigned N = 128;
    const size_t io_size = 4096;
    const size_t total = N * io_size;
    size_t n_words = io_size / sizeof(uint32_t);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, total);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, total, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    // Sync-write 128 distinct patterns
    for (unsigned i = 0; i < N; i++) {
        uint32_t pattern = 0xDE000000 | i;
        fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
            (uint32_t*)((uint8_t*)d_buf + i * io_size), pattern, n_words);
    }
    cudaDeviceSynchronize();

    for (unsigned i = 0; i < N; i++) {
        ssize_t ret = uGDSWrite(fh, d_buf, io_size, i * io_size, i * io_size);
        if (ret < 0) TEST_FAIL("sync write[%u] returned %zd", i, ret);
        if ((size_t)ret != io_size)
            TEST_FAIL("sync write[%u] short: %zd", i, ret);
    }

    // Clear GPU buffer
    cudaMemset(d_buf, 0, total);

    // Batch-read all 128 in one submit
    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, N);
    ASSERT_OK(st, "BatchIOSetUp");

    uGDSIOParams_t params[N];
    for (unsigned i = 0; i < N; i++) {
        memset(&params[i], 0, sizeof(params[i]));
        params[i].devPtr_base = d_buf;
        params[i].file_offset = i * io_size;
        params[i].devPtr_offset = i * io_size;
        params[i].size = io_size;
        params[i].opcode = UGDS_READ;
        params[i].cookie = (void*)(uintptr_t)(i + 1);
    }

    st = uGDSBatchIOSubmit(batch, N, params, 0);
    ASSERT_OK(st, "BatchIOSubmit");

    uGDSIOEvents_t events[N];
    unsigned nr_out = 0;
    st = uGDSBatchIOGetStatus(batch, N, &nr_out, events, nullptr);
    ASSERT_OK(st, "BatchIOGetStatus");

    if (nr_out != N)
        TEST_FAIL("expected %u events, got %u", N, nr_out);

    for (unsigned i = 0; i < nr_out; i++) {
        if (events[i].status != UGDS_BATCH_COMPLETE)
            TEST_FAIL("event[%u] status 0x%x", i, events[i].status);
        if (events[i].ret != (ssize_t)io_size)
            TEST_FAIL("event[%u] ret %zd", i, events[i].ret);
    }

    // Verify all 128 patterns
    uint8_t* h_buf = (uint8_t*)malloc(total);
    cudaMemcpy(h_buf, d_buf, total, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < N; i++) {
        uint32_t expected = 0xDE000000 | i;
        uint32_t* words = (uint32_t*)(h_buf + i * io_size);
        for (size_t w = 0; w < n_words; w++) {
            if (words[w] != expected) {
                free(h_buf);
                TEST_FAIL("region[%u] word %zu: 0x%08X != 0x%08X",
                          i, w, words[w], expected);
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
