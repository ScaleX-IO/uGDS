#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t io_size = 4096;
    const uint32_t write_pattern = 0xBA7C4001;
    const size_t alloc_size = 65536;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, 0);
    ASSERT_OK(st, "BufRegister");

    // Step 1: sync-write a known pattern to offset 0
    size_t n_words = io_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, write_pattern, n_words);
    cudaDeviceSynchronize();

    ssize_t ret = uGDSWrite(fh, d_buf, io_size, 0, 0);
    if (ret < 0) TEST_FAIL("sync write returned %zd", ret);
    if ((size_t)ret != io_size) TEST_FAIL("sync write short: %zd / %zu", ret, io_size);

    // Clear GPU buffer
    cudaMemset(d_buf, 0, alloc_size);

    // Step 2: batch-read it back
    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, 4);
    ASSERT_OK(st, "BatchIOSetUp");

    uGDSIOParams_t params;
    memset(&params, 0, sizeof(params));
    params.devPtr_base = d_buf;
    params.file_offset = 0;
    params.devPtr_offset = 0;
    params.size = io_size;
    params.opcode = UGDS_READ;
    params.cookie = (void*)0x1234;

    st = uGDSBatchIOSubmit(batch, 1, &params, 0);
    ASSERT_OK(st, "BatchIOSubmit");

    // Step 3: poll for completion
    unsigned nr_out = 0;
    uGDSIOEvents_t events[4];
    st = uGDSBatchIOGetStatus(batch, 1, &nr_out, events, nullptr);
    ASSERT_OK(st, "BatchIOGetStatus");

    if (nr_out != 1)
        TEST_FAIL("expected 1 event, got %u", nr_out);
    if (events[0].status != UGDS_BATCH_COMPLETE)
        TEST_FAIL("event status: expected COMPLETE(0x%x), got 0x%x",
                  UGDS_BATCH_COMPLETE, events[0].status);
    if (events[0].ret != (ssize_t)io_size)
        TEST_FAIL("event ret: expected %zu, got %zd", io_size, events[0].ret);
    if (events[0].cookie != (void*)0x1234)
        TEST_FAIL("event cookie mismatch");

    // Step 4: verify data on host
    uint32_t* h_buf = (uint32_t*)malloc(io_size);
    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != write_pattern) {
            free(h_buf);
            TEST_FAIL("data mismatch at word %zu: 0x%08X != 0x%08X",
                      i, h_buf[i], write_pattern);
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
