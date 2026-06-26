#include "test_utils.h"

int main(int argc, char** argv)
{
    parse_args(argc, argv);
    cudaSetDevice(g_gpu_id);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t size = 4096;
    off_t file_off = 0;
    off_t buf_off = 0;
    ssize_t result = 0;
    void* dummy_buf = (void*)0x1;

    // Test 1: driver not initialized
    {
        uGDSError_t st = uGDSReadAsync(nullptr, dummy_buf, &size,
                                         &file_off, &buf_off, &result, stream);
        ASSERT_ERR(st, UGDS_DRIVER_NOT_INITIALIZED, "driver not initialized");
    }

    ASSERT_OK(uGDSDriverOpen(), "DriverOpen");
    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    // Test 2: NULL size_p
    {
        uGDSError_t st = uGDSReadAsync(fh, dummy_buf, nullptr,
                                         &file_off, &buf_off, &result, stream);
        ASSERT_ERR(st, UGDS_INVALID_VALUE, "NULL size_p");
    }

    // Test 3: NULL file_offset_p
    {
        uGDSError_t st = uGDSReadAsync(fh, dummy_buf, &size,
                                         nullptr, &buf_off, &result, stream);
        ASSERT_ERR(st, UGDS_INVALID_VALUE, "NULL file_offset_p");
    }

    // Test 4: NULL bufPtr_offset_p
    {
        uGDSError_t st = uGDSReadAsync(fh, dummy_buf, &size,
                                         &file_off, nullptr, &result, stream);
        ASSERT_ERR(st, UGDS_INVALID_VALUE, "NULL bufPtr_offset_p");
    }

    // Test 5: NULL bytes_read_p
    {
        uGDSError_t st = uGDSReadAsync(fh, dummy_buf, &size,
                                         &file_off, &buf_off, nullptr, stream);
        ASSERT_ERR(st, UGDS_INVALID_VALUE, "NULL bytes_read_p");
    }

    // Test 6: NULL handle
    {
        uGDSError_t st = uGDSReadAsync(nullptr, dummy_buf, &size,
                                         &file_off, &buf_off, &result, stream);
        ASSERT_ERR(st, UGDS_INVALID_VALUE, "NULL handle");
    }

    // Test 7: NULL buffer pointer
    {
        uGDSError_t st = uGDSReadAsync(fh, nullptr, &size,
                                         &file_off, &buf_off, &result, stream);
        ASSERT_ERR(st, UGDS_INVALID_VALUE, "NULL buffer");
    }

    // Test 8: default stream (cudaStream_t = 0) should work
    {
        void* d_buf;
        cudaMalloc(&d_buf, 65536);
        ASSERT_OK(uGDSBufRegister(d_buf, 65536, 0), "BufRegister");

        ssize_t bytes = 0;
        uGDSError_t st = uGDSReadAsync(fh, d_buf, &size, &file_off, &buf_off,
                                         &bytes, 0);
        ASSERT_OK(st, "default stream enqueue");
        cudaStreamSynchronize(0);

        uGDSBufDeregister(d_buf);
        cudaFree(d_buf);
    }

    // Test 9: StreamRegister / StreamDeregister basic
    {
        ASSERT_OK(uGDSStreamRegister(stream), "StreamRegister");
        ASSERT_OK(uGDSStreamDeregister(stream), "StreamDeregister");
    }

    cudaStreamDestroy(stream);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}
