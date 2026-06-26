#include "test_utils.h"

__global__ void write_size_kernel(size_t* size_p, size_t value)
{
    *size_p = value;
}

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
    ASSERT_OK(uGDSBufRegister(d_buf, alloc_size, 0), "BufRegister");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Fill buffer with pattern
    uint32_t pattern = 0xBEEFCAFE;
    size_t n_words = io_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256, 0, stream>>>(
        (uint32_t*)d_buf, pattern, n_words);

    // Allocate size parameter in pinned memory (GPU-writable, CPU-readable)
    size_t* size_p;
    cudaHostAlloc(&size_p, sizeof(size_t), cudaHostAllocDefault);
    *size_p = 0;  // initially zero — NOT the real IO size

    off_t file_off = 0;
    off_t buf_off = 0;
    ssize_t bytes_written = 0;

    // Launch kernel on stream that sets *size_p = io_size
    // This kernel runs BEFORE the async IO callback
    write_size_kernel<<<1, 1, 0, stream>>>(size_p, io_size);

    // Enqueue async write — at this point *size_p is still 0 on the CPU
    // but the kernel will set it to io_size before the callback runs
    ASSERT_OK(uGDSWriteAsync(fh, d_buf, size_p, &file_off, &buf_off,
                               &bytes_written, stream), "WriteAsync");

    cudaStreamSynchronize(stream);

    // If late binding works, bytes_written == io_size (from kernel-written *size_p)
    // If eagerly evaluated, bytes_written would be 0 or error
    if (bytes_written != (ssize_t)io_size)
        TEST_FAIL("late binding failed: expected %zu, got %zd", io_size, bytes_written);

    // Verify: read back and check pattern
    cudaMemsetAsync(d_buf, 0, alloc_size, stream);
    size_t read_size = io_size;
    ssize_t bytes_read = 0;
    ASSERT_OK(uGDSReadAsync(fh, d_buf, &read_size, &file_off, &buf_off,
                              &bytes_read, stream), "ReadAsync");
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

    cudaFreeHost(size_p);
    cudaStreamDestroy(stream);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}
