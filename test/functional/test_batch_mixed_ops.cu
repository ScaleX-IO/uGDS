#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t io_size = 4096;
    const uint32_t patternA = 0xAAAA1111;
    const uint32_t patternB = 0xBBBB2222;
    const size_t alloc_size = 65536;
    size_t n_words = io_size / sizeof(uint32_t);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, 4);
    ASSERT_OK(st, "BatchIOSetUp");

    // Round 1: batch-write two patterns to disk
    uint8_t* d_off0 = (uint8_t*)d_buf;
    uint8_t* d_off1 = (uint8_t*)d_buf + io_size;

    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_off0, patternA, n_words);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_off1, patternB, n_words);
    cudaDeviceSynchronize();

    uGDSIOParams_t wp[2];
    memset(wp, 0, sizeof(wp));
    wp[0].devPtr_base = d_buf;
    wp[0].file_offset = 0;
    wp[0].devPtr_offset = 0;
    wp[0].size = io_size;
    wp[0].opcode = UGDS_WRITE;
    wp[0].cookie = (void*)(uintptr_t)0x1001;

    wp[1].devPtr_base = d_buf;
    wp[1].file_offset = io_size;
    wp[1].devPtr_offset = io_size;
    wp[1].size = io_size;
    wp[1].opcode = UGDS_WRITE;
    wp[1].cookie = (void*)(uintptr_t)0x1002;

    st = uGDSBatchIOSubmit(batch, 2, wp, 0);
    ASSERT_OK(st, "Write Submit");

    unsigned nr_out = 0;
    uGDSIOEvents_t events[4];
    st = uGDSBatchIOGetStatus(batch, 2, &nr_out, events, nullptr);
    ASSERT_OK(st, "Write GetStatus");
    if (nr_out != 2) TEST_FAIL("write: expected 2 events, got %u", nr_out);
    for (unsigned i = 0; i < nr_out; i++) {
        if (events[i].status != UGDS_BATCH_COMPLETE)
            TEST_FAIL("write event[%u] status 0x%x", i, events[i].status);
    }

    // Clear GPU buffer before read-back
    cudaMemset(d_buf, 0, alloc_size);

    // Round 2: batch-read both back
    uGDSIOParams_t rp[2];
    memset(rp, 0, sizeof(rp));
    rp[0].devPtr_base = d_buf;
    rp[0].file_offset = 0;
    rp[0].devPtr_offset = 0;
    rp[0].size = io_size;
    rp[0].opcode = UGDS_READ;
    rp[0].cookie = (void*)(uintptr_t)0x2001;

    rp[1].devPtr_base = d_buf;
    rp[1].file_offset = io_size;
    rp[1].devPtr_offset = io_size;
    rp[1].size = io_size;
    rp[1].opcode = UGDS_READ;
    rp[1].cookie = (void*)(uintptr_t)0x2002;

    st = uGDSBatchIOSubmit(batch, 2, rp, 0);
    ASSERT_OK(st, "Read Submit");

    st = uGDSBatchIOGetStatus(batch, 2, &nr_out, events, nullptr);
    ASSERT_OK(st, "Read GetStatus");
    if (nr_out != 2) TEST_FAIL("read: expected 2 events, got %u", nr_out);
    for (unsigned i = 0; i < nr_out; i++) {
        if (events[i].status != UGDS_BATCH_COMPLETE)
            TEST_FAIL("read event[%u] status 0x%x", i, events[i].status);
    }

    // Verify data
    uint32_t* h_buf = (uint32_t*)malloc(2 * io_size);
    cudaMemcpy(h_buf, d_buf, 2 * io_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != patternA)
            TEST_FAIL("region0 mismatch at %zu: 0x%08X != 0x%08X", i, h_buf[i], patternA);
    }
    uint32_t* h_off1 = h_buf + n_words;
    for (size_t i = 0; i < n_words; i++) {
        if (h_off1[i] != patternB)
            TEST_FAIL("region1 mismatch at %zu: 0x%08X != 0x%08X", i, h_off1[i], patternB);
    }
    free(h_buf);

    uGDSBatchIODestroy(batch);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
