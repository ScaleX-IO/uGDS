#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t sizes[] = {4096, 8192, 65536, 262144};
    const uint32_t patterns[] = {0xAAAA0001, 0xBBBB0002, 0xCCCC0003, 0xDDDD0004};
    const int N = 4;

    size_t offsets[N];
    offsets[0] = 0;
    for (int i = 1; i < N; i++)
        offsets[i] = offsets[i - 1] + sizes[i - 1];

    size_t total = offsets[N - 1] + sizes[N - 1];

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, total);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, total, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    // Step 1: sync-write each region with a distinct pattern
    for (int i = 0; i < N; i++) {
        size_t n_words = sizes[i] / sizeof(uint32_t);
        uint8_t* ptr = (uint8_t*)d_buf + offsets[i];
        fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
            (uint32_t*)ptr, patterns[i], n_words);
        cudaDeviceSynchronize();

        ssize_t ret = uGDSWrite(fh, d_buf, sizes[i], offsets[i], offsets[i]);
        if (ret < 0) TEST_FAIL("sync write[%d] returned %zd", i, ret);
        if ((size_t)ret != sizes[i])
            TEST_FAIL("sync write[%d] short: %zd / %zu", i, ret, sizes[i]);
    }

    // Clear GPU buffer
    cudaMemset(d_buf, 0, total);
    cudaDeviceSynchronize();

    // Step 2: batch-read all 4 regions
    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, N);
    ASSERT_OK(st, "BatchIOSetUp");

    uGDSIOParams_t params[N];
    for (int i = 0; i < N; i++) {
        memset(&params[i], 0, sizeof(params[i]));
        params[i].devPtr_base = d_buf;
        params[i].file_offset = offsets[i];
        params[i].devPtr_offset = offsets[i];
        params[i].size = sizes[i];
        params[i].opcode = UGDS_READ;
        params[i].cookie = (void*)(uintptr_t)(i + 1);
    }

    st = uGDSBatchIOSubmit(batch, N, params, 0);
    ASSERT_OK(st, "BatchIOSubmit");

    // Step 3: poll for all completions
    unsigned nr_out = 0;
    uGDSIOEvents_t events[N];
    st = uGDSBatchIOGetStatus(batch, N, &nr_out, events, nullptr);
    ASSERT_OK(st, "BatchIOGetStatus");

    if (nr_out != N)
        TEST_FAIL("expected %d events, got %u", N, nr_out);

    for (unsigned e = 0; e < nr_out; e++) {
        if (events[e].status != UGDS_BATCH_COMPLETE)
            TEST_FAIL("event[%u] status 0x%x != COMPLETE", e, events[e].status);
    }

    // Step 4: verify data
    uint8_t* h_buf = (uint8_t*)malloc(total);
    cudaMemcpy(h_buf, d_buf, total, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        uint32_t* words = (uint32_t*)(h_buf + offsets[i]);
        size_t n_words = sizes[i] / sizeof(uint32_t);
        for (size_t w = 0; w < n_words; w++) {
            if (words[w] != patterns[i]) {
                free(h_buf);
                TEST_FAIL("region[%d] mismatch at word %zu: 0x%08X != 0x%08X",
                          i, w, words[w], patterns[i]);
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
