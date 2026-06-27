#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#ifdef USE_NVIDIA_GDS
#include <cufile.h>
#define uGDSError_t       CUfileError_t
#define uGDSHandle_t      CUfileHandle_t
#define uGDSDescr_t       CUfileDescr_t
#define uGDSDriverOpen    cuFileDriverOpen
#define uGDSDriverClose   cuFileDriverClose
#define uGDSHandleRegister   cuFileHandleRegister
#define uGDSHandleDeregister cuFileHandleDeregister
#define uGDSBufRegister   cuFileBufRegister
#define uGDSBufDeregister cuFileBufDeregister
#define uGDSRead          cuFileRead
#define uGDSWrite         cuFileWrite
#define uGDSReadAsync     cuFileReadAsync
#define uGDSWriteAsync    cuFileWriteAsync
#define uGDSStreamDeregister cuFileStreamDeregister
#define UGDS_SUCCESS      CU_FILE_SUCCESS
#define UGDS_HANDLE_TYPE_OPAQUE_FD  CU_FILE_HANDLE_TYPE_OPAQUE_FD
static inline uGDSError_t uGDSStreamRegister(cudaStream_t s) { return cuFileStreamRegister(s, 0); }
#else
#include <ugds.h>
#endif

#include "bench_utils.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void busy_kernel(float* out, size_t n, int iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = (float)idx;
    for (int i = 0; i < iters; i++)
        v = v * 1.0001f + 0.0001f;
    out[idx] = v;
}

static uint64_t ts_ns(const struct timespec& a, const struct timespec& b) {
    return (b.tv_sec - a.tv_sec) * 1000000000ULL + (b.tv_nsec - a.tv_nsec);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device_or_file> [gpu_id] [io_size] [num_ios] [kernel_iters]\n", argv[0]);
        return 1;
    }
    const char* dev_path = argv[1];
    int gpu_id = argc > 2 ? atoi(argv[2]) : 0;
    size_t io_size = argc > 3 ? parse_size(argv[3]) : 1 * MB;
    int num_ios = argc > 4 ? atoi(argv[4]) : 16;
    int kernel_iters = argc > 5 ? atoi(argv[5]) : 5000;

    CHECK_CUDA(cudaSetDevice(gpu_id));

    uGDSError_t status = uGDSDriverOpen();
    if (status.err != UGDS_SUCCESS) { fprintf(stderr, "DriverOpen failed\n"); return 1; }

    int open_flags = O_RDWR;
#ifdef USE_O_DIRECT
    open_flags |= O_DIRECT;
#endif
    int fd = open(dev_path, open_flags, 0644);
    if (fd < 0) { perror("open"); return 1; }

    uGDSDescr_t descr;
    uGDSHandle_t fh;
    memset(&descr, 0, sizeof(descr));
    descr.handle.fd = fd;
    descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    status = uGDSHandleRegister(&fh, &descr);
    if (status.err != UGDS_SUCCESS) { fprintf(stderr, "HandleRegister failed\n"); return 1; }

    size_t total_buf = io_size * num_ios;
    void* d_io_buf;
    CHECK_CUDA(cudaMalloc(&d_io_buf, total_buf));
    CHECK_CUDA(cudaMemset(d_io_buf, 0xAB, total_buf));
    status = uGDSBufRegister(d_io_buf, total_buf, 0);
    if (status.err != UGDS_SUCCESS) { fprintf(stderr, "BufRegister failed\n"); return 1; }

    size_t compute_n = 256 * 1024;
    float* d_compute;
    CHECK_CUDA(cudaMalloc(&d_compute, compute_n * sizeof(float)));

    dim3 block(256);
    dim3 grid((compute_n + 255) / 256);

    struct timespec t0, t1;

#ifdef USE_NVIDIA_GDS
    const char* tag = "GDS";
#else
    const char* tag = "uGDS";
#endif

    // ── Baseline: sequential (sync IO then kernel, no overlap) ──
    CHECK_CUDA(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < num_ios; i++) {
        ssize_t ret = uGDSWrite(fh, d_io_buf, io_size,
                                (off_t)(i * io_size), i * io_size);
        if (ret != (ssize_t)io_size) {
            fprintf(stderr, "sync write %d failed: %zd\n", i, ret);
            break;
        }
        busy_kernel<<<grid, block>>>(d_compute, compute_n, kernel_iters);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t seq_ns = ts_ns(t0, t1);

    // ── Shared pinned arrays for overlap and pipeline sections ──
    cudaStream_t stream_io, stream_compute;
    CHECK_CUDA(cudaStreamCreate(&stream_io));
    CHECK_CUDA(cudaStreamCreate(&stream_compute));

    uGDSStreamRegister(stream_io);

    size_t* sizes;
    off_t* file_offs;
    off_t* buf_offs;
    ssize_t* results;
    CHECK_CUDA(cudaHostAlloc(&sizes, num_ios * sizeof(size_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&file_offs, num_ios * sizeof(off_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&buf_offs, num_ios * sizeof(off_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&results, num_ios * sizeof(ssize_t), cudaHostAllocDefault));

    // ── Overlap: async IO on stream_io, kernel on stream_compute ──
    CHECK_CUDA(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < num_ios; i++) {
        sizes[0] = io_size;
        file_offs[0] = (off_t)(i * io_size);
        buf_offs[0] = (off_t)(i * io_size);
        results[0] = 0;

        uGDSError_t ast = uGDSWriteAsync(fh, d_io_buf, &sizes[0],
                                          &file_offs[0], &buf_offs[0],
                                          &results[0], stream_io);
        if (ast.err != UGDS_SUCCESS) {
            fprintf(stderr, "async write %d enqueue failed\n", i);
            break;
        }

        busy_kernel<<<grid, block, 0, stream_compute>>>(
            d_compute, compute_n, kernel_iters);

        CHECK_CUDA(cudaStreamSynchronize(stream_io));
        CHECK_CUDA(cudaStreamSynchronize(stream_compute));

        if (results[0] != (ssize_t)io_size) {
            fprintf(stderr, "async write %d result: %zd\n", i, results[0]);
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t overlap_ns = ts_ns(t0, t1);

    // ── Pipeline: async IO on stream_io, kernel on stream_compute, no per-iter sync ──
    CHECK_CUDA(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < num_ios; i++) {
        sizes[i] = io_size;
        file_offs[i] = (off_t)(i * io_size);
        buf_offs[i] = (off_t)(i * io_size);
        results[i] = 0;

        uGDSWriteAsync(fh, d_io_buf, &sizes[i], &file_offs[i],
                        &buf_offs[i], &results[i], stream_io);

        busy_kernel<<<grid, block, 0, stream_compute>>>(
            d_compute, compute_n, kernel_iters);
    }

    CHECK_CUDA(cudaStreamSynchronize(stream_io));
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));

    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t pipeline_ns = ts_ns(t0, t1);

    // ── Report ──
    printf("[%s] Overlap Benchmark: %d x %zuKB writes + GPU compute (kernel_iters=%d)\n\n",
           tag, num_ios, io_size / 1024, kernel_iters);
    printf("  Sequential (sync IO + kernel):   %8.2f ms\n", seq_ns / 1e6);
    printf("  Overlap (async IO || kernel):    %8.2f ms\n", overlap_ns / 1e6);
    printf("  Pipeline (all async, sync end):  %8.2f ms\n", pipeline_ns / 1e6);
    printf("\n");
    printf("  Overlap speedup vs sequential:   %.2fx\n", (double)seq_ns / overlap_ns);
    printf("  Pipeline speedup vs sequential:  %.2fx\n", (double)seq_ns / pipeline_ns);

    cudaFreeHost(sizes);
    cudaFreeHost(file_offs);
    cudaFreeHost(buf_offs);
    cudaFreeHost(results);
    uGDSStreamDeregister(stream_io);
    cudaStreamDestroy(stream_io);
    cudaStreamDestroy(stream_compute);
    uGDSBufDeregister(d_io_buf);
    cudaFree(d_io_buf);
    cudaFree(d_compute);
    uGDSHandleDeregister(fh);
    close(fd);
    uGDSDriverClose();
    return 0;
}
