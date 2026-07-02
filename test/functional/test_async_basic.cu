#include "test_utils.h"

int main(int argc, char** argv)
{
    parse_args(argc, argv);
    cudaSetDevice(g_gpu_id);

    ASSERT_OK(uGDSDriverOpen(), "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t alloc_size = 65536;
    const size_t io_size = 4096;
    void* d_buf;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    ASSERT_OK(uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS), "BufRegister");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint32_t pattern = 0xCAFEBABE;
    size_t n_words = io_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256, 0, stream>>>(
        (uint32_t*)d_buf, pattern, n_words);

    size_t size = io_size;
    off_t file_off = 0;
    off_t buf_off = 0;
    ssize_t bytes_written = 0;

    uGDSError_t st = uGDSWriteAsync(fh, d_buf, &size, &file_off, &buf_off,
                                      &bytes_written, stream);
    ASSERT_OK(st, "WriteAsync");
    cudaStreamSynchronize(stream);

    if (bytes_written != (ssize_t)io_size)
        TEST_FAIL("WriteAsync: expected %zu, got %zd", io_size, bytes_written);

    cudaMemsetAsync(d_buf, 0, alloc_size, stream);

    ssize_t bytes_read = 0;
    st = uGDSReadAsync(fh, d_buf, &size, &file_off, &buf_off,
                        &bytes_read, stream);
    ASSERT_OK(st, "ReadAsync");
    cudaStreamSynchronize(stream);

    if (bytes_read != (ssize_t)io_size)
        TEST_FAIL("ReadAsync: expected %zu, got %zd", io_size, bytes_read);

    uint32_t h_buf[io_size / sizeof(uint32_t)];
    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != pattern)
            TEST_FAIL("data mismatch at word %zu: 0x%08X != 0x%08X",
                       i, h_buf[i], pattern);
    }

    cudaStreamDestroy(stream);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}
