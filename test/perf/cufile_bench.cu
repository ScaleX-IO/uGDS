#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>

#ifdef USE_NVIDIA_GDS
#include <cufile.h>
#define uGDSError_t       CUfileError_t
#define uGDSHandle_t      CUfileHandle_t
#define uGDSDescr_t       CUfileDescr_t
#define uGDSOpError       CUfileOpError
#define uGDSDriverOpen    cuFileDriverOpen
#define uGDSDriverClose   cuFileDriverClose
#define uGDSHandleRegister   cuFileHandleRegister
#define uGDSHandleDeregister cuFileHandleDeregister
#define uGDSBufRegister   cuFileBufRegister
#define uGDSBufDeregister cuFileBufDeregister
#define uGDSRead          cuFileRead
#define uGDSWrite         cuFileWrite
#define UGDS_SUCCESS      CU_FILE_SUCCESS
#define UGDS_HANDLE_TYPE_OPAQUE_FD  CU_FILE_HANDLE_TYPE_OPAQUE_FD
#define uGDS_status_error cufileop_status_error
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

#define CHECK_UGDS(call, msg)                                                \
    do {                                                                        \
        uGDSError_t status = (call);                                          \
        if (status.err != UGDS_SUCCESS) {                                    \
            fprintf(stderr, "%s failed: %s\n", msg,                             \
                    uGDS_status_error(status.err));                          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static void* sync_rw_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    struct timespec io_start, io_end;
    uGDSHandle_t cf_handle = *(uGDSHandle_t*)data->handler;
    ssize_t io_size = (ssize_t)data->io_size;
    size_t done_bytes = 0;

    clock_gettime(CLOCK_MONOTONIC, &data->start_time);

    while (done_bytes < data->size) {
        clock_gettime(CLOCK_MONOTONIC, &io_start);

        ssize_t result;
        if (data->mode == 1) {
            result = uGDSWrite(cf_handle, data->gpu_buffer, data->io_size,
                                 data->offset + done_bytes, done_bytes);
        } else {
            result = uGDSRead(cf_handle, data->gpu_buffer, data->io_size,
                                data->offset + done_bytes, 0);
        }

        if (result == 0) {
            break;
        }
        if (result != io_size) {
            fprintf(stderr, "thread %d: IO error, result=%zd, expected=%zd\n",
                    data->thread_id, result, io_size);
            return NULL;
        }

        clock_gettime(CLOCK_MONOTONIC, &io_end);

        uint64_t io_time = ts_diff_ns(io_start, io_end);
        data->latency_vec.push_back(io_time);
        data->total_io_time += io_time;
        data->io_operations++;
        done_bytes += result;
    }

    clock_gettime(CLOCK_MONOTONIC, &data->end_time);
    return NULL;
}

static void run_ugds_bench(BenchOpts& opts) {
    struct timespec prog_start, prog_end;
    uGDSError_t status;
    uGDSDescr_t cf_descr;
    uGDSHandle_t cf_handle;
    int file_fd;

    const char* label = opts.is_write ? "uGDS-write" : "uGDS-read";

    printf("=== uGDS Benchmark ===\n");
    printf("  File:       %s\n", opts.file_path);
    printf("  Length:     %zu bytes (%.2f MB)\n", opts.length, opts.length / (double)MB);
    printf("  IO size:    %zu bytes\n", opts.io_size);
    printf("  Threads:    %d\n", opts.num_threads);
    printf("  IO depth:   %d\n", opts.io_depth);
    printf("  GPU:        %d\n", opts.gpu_id);
    printf("  Mode:       %s\n", opts.is_write ? "write" : "read");
    printf("\n");

    CHECK_CUDA(cudaSetDevice(opts.gpu_id));

    status = uGDSDriverOpen();
    if (status.err != UGDS_SUCCESS) {
        fprintf(stderr, "uGDSDriverOpen failed: %s\n",
                uGDS_status_error(status.err));
        exit(EXIT_FAILURE);
    }

    int open_flags = O_RDWR;
#ifdef USE_O_DIRECT
    open_flags |= O_DIRECT;
#endif
    file_fd = open(opts.file_path, open_flags, 0644);
    if (file_fd < 0) {
        perror("open file failed");
        uGDSDriverClose();
        exit(EXIT_FAILURE);
    }

    memset(&cf_descr, 0, sizeof(uGDSDescr_t));
    cf_descr.handle.fd = file_fd;
    cf_descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    CHECK_UGDS(uGDSHandleRegister(&cf_handle, &cf_descr), "uGDSHandleRegister");

    size_t chunk_size = opts.length / opts.num_threads;
    int num_threads = opts.num_threads;

    std::vector<ThreadData> threads(num_threads);
    std::vector<pthread_t> pthreads(num_threads);

    for (int i = 0; i < num_threads; i++) {
        ThreadData* td = &threads[i];
        td->thread_id = i;
        td->fd = file_fd;
        td->mode = opts.is_write ? 1 : 0;
        td->offset = (off_t)(i * chunk_size);
        td->size = chunk_size;
        td->io_size = opts.io_size;
        td->depth = opts.io_depth;
        td->handler = &cf_handle;
        td->total_io_time = 0;
        td->io_operations = 0;
        td->device_id = opts.gpu_id;

        CHECK_CUDA(cudaMalloc(&td->gpu_buffer, chunk_size));
        CHECK_CUDA(cudaMemset(td->gpu_buffer, 0x00, chunk_size));
        CHECK_CUDA(cudaStreamSynchronize(0));

        CHECK_UGDS(uGDSBufRegister(td->gpu_buffer, chunk_size, 0),
                     "uGDSBufRegister");

        td->latency_vec.reserve(chunk_size / opts.io_size + 10);
    }

    clock_gettime(CLOCK_MONOTONIC, &prog_start);

    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pthreads[i], NULL, sync_rw_thread, &threads[i]) != 0) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &prog_end);

    uint64_t prog_time_ns = ts_diff_ns(prog_start, prog_end);

    report_results(label, threads, opts.io_size, opts.io_depth, prog_time_ns, opts.json);

    for (int i = 0; i < num_threads; i++) {
        uGDSBufDeregister(threads[i].gpu_buffer);
        CHECK_CUDA(cudaFree(threads[i].gpu_buffer));
    }

    uGDSHandleDeregister(cf_handle);
    close(file_fd);
    uGDSDriverClose();
}

int main(int argc, char** argv) {
    BenchOpts opts;
    if (!parse_bench_opts(argc, argv, opts)) {
        return EXIT_FAILURE;
    }
    run_ugds_bench(opts);
    return 0;
}
