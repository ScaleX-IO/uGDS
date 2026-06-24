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
#define uGDSBatchHandle_t    CUfileBatchHandle_t
#define uGDSIOEvents_t       CUfileIOEvents_t
#define uGDSBatchIOGetStatus cuFileBatchIOGetStatus
#define uGDSBatchIODestroy   cuFileBatchIODestroy
#define UGDS_BATCH_COMPLETE  CU_FILE_BATCH_IO_COMPLETE
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

static void* batch_rw_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    struct timespec batch_start, batch_end;
    uGDSHandle_t cf_handle = *(uGDSHandle_t*)data->handler;
    size_t batch_size = data->depth;
    size_t done_bytes = 0;

    uGDSBatchHandle_t batch = nullptr;
#ifdef USE_NVIDIA_GDS
    uGDSError_t bst = cuFileBatchIOSetUp(&batch, static_cast<unsigned>(batch_size));
#else
    uGDSError_t bst = uGDSBatchIOSetUp(&batch, cf_handle,
        static_cast<unsigned>(batch_size));
#endif
    if (bst.err != UGDS_SUCCESS) {
        fprintf(stderr, "thread %d: BatchIOSetUp failed\n", data->thread_id);
        return nullptr;
    }

#ifdef USE_NVIDIA_GDS
    std::vector<CUfileIOParams_t> params(batch_size);
#else
    std::vector<uGDSIOParams_t> params(batch_size);
#endif
    std::vector<uGDSIOEvents_t> events(batch_size);

    clock_gettime(CLOCK_MONOTONIC, &data->start_time);

    while (done_bytes < data->size) {
        unsigned nr = 0;
        for (size_t i = 0; i < batch_size && done_bytes + i * data->io_size < data->size; i++) {
            memset(&params[i], 0, sizeof(params[0]));
#ifdef USE_NVIDIA_GDS
            params[i].mode = CUFILE_BATCH;
            params[i].u.batch.devPtr_base = data->gpu_buffer;
            params[i].u.batch.file_offset = data->offset + done_bytes + i * data->io_size;
            params[i].u.batch.devPtr_offset = (done_bytes + i * data->io_size) % data->size;
            params[i].u.batch.size = data->io_size;
            params[i].fh = cf_handle;
            params[i].opcode = data->mode ? CUFILE_WRITE : CUFILE_READ;
#else
            params[i].devPtr_base = data->gpu_buffer;
            params[i].file_offset = data->offset + done_bytes + i * data->io_size;
            params[i].devPtr_offset = (done_bytes + i * data->io_size) % data->size;
            params[i].size = data->io_size;
            params[i].opcode = data->mode ? UGDS_WRITE : UGDS_READ;
#endif
            nr++;
        }
        if (nr == 0) break;

        clock_gettime(CLOCK_MONOTONIC, &batch_start);

#ifdef USE_NVIDIA_GDS
        uGDSError_t st = cuFileBatchIOSubmit(batch, nr, params.data(), 0);
#else
        uGDSError_t st = uGDSBatchIOSubmit(batch, nr, params.data(), 0);
#endif
        if (st.err != UGDS_SUCCESS) {
            fprintf(stderr, "thread %d: BatchIOSubmit failed\n", data->thread_id);
            break;
        }

        unsigned nr_out = nr;
        st = uGDSBatchIOGetStatus(batch, nr, &nr_out, events.data(), nullptr);
        if (st.err != UGDS_SUCCESS) {
            fprintf(stderr, "thread %d: BatchIOGetStatus failed\n", data->thread_id);
            break;
        }

        clock_gettime(CLOCK_MONOTONIC, &batch_end);

        uint64_t batch_time = ts_diff_ns(batch_start, batch_end);
        data->latency_vec.push_back(batch_time);
        data->total_io_time += batch_time;
        data->io_operations += nr;
        done_bytes += nr * data->io_size;
    }

    clock_gettime(CLOCK_MONOTONIC, &data->end_time);
    uGDSBatchIODestroy(batch);
    return nullptr;
}

static void run_ugds_bench(BenchOpts& opts) {
    struct timespec prog_start, prog_end;
    uGDSError_t status;
    uGDSDescr_t cf_descr;
    uGDSHandle_t cf_handle;
    int file_fd;

    const char* label;
#ifdef USE_NVIDIA_GDS
    if (opts.is_batch)
        label = opts.is_write ? "GDS-batch-write" : "GDS-batch-read";
    else
        label = opts.is_write ? "GDS-write" : "GDS-read";
#else
    if (opts.is_batch)
        label = opts.is_write ? "uGDS-batch-write" : "uGDS-batch-read";
    else
        label = opts.is_write ? "uGDS-write" : "uGDS-read";
#endif

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
        void* (*thread_fn)(void*) = sync_rw_thread;
        if (opts.is_batch) thread_fn = batch_rw_thread;
        if (pthread_create(&pthreads[i], NULL, thread_fn, &threads[i]) != 0) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &prog_end);

    uint64_t prog_time_ns = ts_diff_ns(prog_start, prog_end);

    report_results(label, threads, opts.io_size, prog_time_ns, opts.json);

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
