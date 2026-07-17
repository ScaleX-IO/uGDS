#include "test_utils.h"
#include <vector>

static const int N_STREAMS = 4;

int main(int argc, char** argv)
{
    parse_args(argc, argv);
    cudaSetDevice(g_gpu_id);

    ASSERT_OK(uGDSDriverOpen(), "DriverOpen");
    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t chunk = 4096;
    const size_t alloc_size = chunk * N_STREAMS;
    void* d_buf;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");
    ASSERT_OK(uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS), "BufRegister");

    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    size_t sizes[N_STREAMS];
    off_t file_offs[N_STREAMS];
    off_t buf_offs[N_STREAMS];
    ssize_t write_results[N_STREAMS];
    ssize_t read_results[N_STREAMS];

    // Phase 1: each stream writes unique pattern to unique file region
    for (int i = 0; i < N_STREAMS; i++) {
        uint32_t pattern = 0xBEEF0000 | (uint32_t)i;
        size_t n_words = chunk / sizeof(uint32_t);
        uint32_t* region = (uint32_t*)((uint8_t*)d_buf + i * chunk);
        fill_pattern_u32<<<(n_words + 255) / 256, 256, 0, streams[i]>>>(
            region, pattern, n_words);

        sizes[i] = chunk;
        file_offs[i] = (off_t)(i * chunk);
        buf_offs[i] = (off_t)(i * chunk);
        write_results[i] = 0;

        ASSERT_OK(uGDSWriteAsync(fh, d_buf, &sizes[i], &file_offs[i],
                                   &buf_offs[i], &write_results[i], streams[i]),
                  "WriteAsync");
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < N_STREAMS; i++) {
        if (write_results[i] != (ssize_t)chunk)
            TEST_FAIL("stream %d write: expected %zu, got %zd",
                       i, chunk, write_results[i]);
    }

    // Phase 2: clear buffer, read back on each stream
    cudaMemset(d_buf, 0, alloc_size);
    cudaDeviceSynchronize();

    for (int i = 0; i < N_STREAMS; i++) {
        read_results[i] = 0;
        ASSERT_OK(uGDSReadAsync(fh, d_buf, &sizes[i], &file_offs[i],
                                  &buf_offs[i], &read_results[i], streams[i]),
                  "ReadAsync");
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < N_STREAMS; i++) {
        if (read_results[i] != (ssize_t)chunk)
            TEST_FAIL("stream %d read: expected %zu, got %zd",
                       i, chunk, read_results[i]);
    }

    // Phase 3: verify each region has correct pattern
    std::vector<uint8_t> h_buf(alloc_size);
    cudaMemcpy(h_buf.data(), d_buf, alloc_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_STREAMS; i++) {
        uint32_t expected = 0xBEEF0000 | (uint32_t)i;
        uint32_t* words = (uint32_t*)(h_buf.data() + i * chunk);
        size_t n_words = chunk / sizeof(uint32_t);
        for (size_t w = 0; w < n_words; w++) {
            if (words[w] != expected)
                TEST_FAIL("stream %d word %zu: 0x%08X != 0x%08X",
                           i, w, words[w], expected);
        }
    }

    for (int i = 0; i < N_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}
