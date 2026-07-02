#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t io_size = 4096;
    const size_t alloc_size = 4 * io_size;
    size_t n_words = io_size / sizeof(uint32_t);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, 4);
    ASSERT_OK(st, "BatchIOSetUp");

    // Round 1: write 4 × 4KB with distinct patterns
    const uint32_t patterns[] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
    for (int i = 0; i < 4; i++) {
        fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
            (uint32_t*)((uint8_t*)d_buf + i * io_size), patterns[i], n_words);
    }
    cudaDeviceSynchronize();

    uGDSIOParams_t params[4];
    for (int i = 0; i < 4; i++) {
        memset(&params[i], 0, sizeof(params[i]));
        params[i].devPtr_base = d_buf;
        params[i].file_offset = i * io_size;
        params[i].devPtr_offset = i * io_size;
        params[i].size = io_size;
        params[i].opcode = UGDS_WRITE;
        params[i].cookie = (void*)(uintptr_t)(0x100 + i);
    }

    st = uGDSBatchIOSubmit(batch, 4, params, 0);
    ASSERT_OK(st, "Round1 Submit");

    unsigned nr_out = 0;
    uGDSIOEvents_t events[4];
    st = uGDSBatchIOGetStatus(batch, 4, &nr_out, events, nullptr);
    ASSERT_OK(st, "Round1 GetStatus");
    if (nr_out != 4) TEST_FAIL("round1: expected 4, got %u", nr_out);

    // Round 2: reuse batch — read back 3 of the 4 regions
    cudaMemset(d_buf, 0, alloc_size);

    for (int i = 0; i < 3; i++) {
        memset(&params[i], 0, sizeof(params[i]));
        params[i].devPtr_base = d_buf;
        params[i].file_offset = i * io_size;
        params[i].devPtr_offset = i * io_size;
        params[i].size = io_size;
        params[i].opcode = UGDS_READ;
        params[i].cookie = (void*)(uintptr_t)(0x200 + i);
    }

    st = uGDSBatchIOSubmit(batch, 3, params, 0);
    ASSERT_OK(st, "Round2 Submit");

    st = uGDSBatchIOGetStatus(batch, 3, &nr_out, events, nullptr);
    ASSERT_OK(st, "Round2 GetStatus");
    if (nr_out != 3) TEST_FAIL("round2: expected 3, got %u", nr_out);

    // Verify first 3 regions
    uint32_t* h_buf = (uint32_t*)malloc(alloc_size);
    cudaMemcpy(h_buf, d_buf, 3 * io_size, cudaMemcpyDeviceToHost);

    for (int r = 0; r < 3; r++) {
        uint32_t* words = h_buf + r * n_words;
        for (size_t w = 0; w < n_words; w++) {
            if (words[w] != patterns[r]) {
                free(h_buf);
                TEST_FAIL("round2 region[%d] word %zu: 0x%08X != 0x%08X",
                          r, w, words[w], patterns[r]);
            }
        }
    }

    // Round 3: reuse again — read the 4th region
    cudaMemset(d_buf, 0, alloc_size);
    memset(&params[0], 0, sizeof(params[0]));
    params[0].devPtr_base = d_buf;
    params[0].file_offset = 3 * io_size;
    params[0].devPtr_offset = 0;
    params[0].size = io_size;
    params[0].opcode = UGDS_READ;
    params[0].cookie = (void*)(uintptr_t)0x300;

    st = uGDSBatchIOSubmit(batch, 1, params, 0);
    ASSERT_OK(st, "Round3 Submit");

    st = uGDSBatchIOGetStatus(batch, 1, &nr_out, events, nullptr);
    ASSERT_OK(st, "Round3 GetStatus");
    if (nr_out != 1) TEST_FAIL("round3: expected 1, got %u", nr_out);
    if (events[0].status != UGDS_BATCH_COMPLETE)
        TEST_FAIL("round3 event status 0x%x", events[0].status);

    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);
    for (size_t w = 0; w < n_words; w++) {
        if (h_buf[w] != patterns[3]) {
            free(h_buf);
            TEST_FAIL("round3 word %zu: 0x%08X != 0x%08X", w, h_buf[w], patterns[3]);
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
