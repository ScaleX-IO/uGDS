/* Scalar vs vectored (scatter-gather) throughput comparison on one
 * registered buffer.  Measures the same total bytes through N discrete
 * scalar uGDSRead/uGDSWrite calls versus one uGDSReadv/uGDSWritev call
 * with N segments, so the command-count reduction is the only variable.
 *
 * The scattered layout matches the typical GDS use case: many small
 * device buffers that would otherwise need one command each.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>

#ifdef __HIP_PLATFORM_AMD__
  #include <hip/hip_runtime.h>
  #define cudaSetDevice        hipSetDevice
  #define cudaMalloc           hipMalloc
  #define cudaFree             hipFree
  #define cudaMemcpy           hipMemcpy
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaGetErrorString   hipGetErrorString
  #define cudaSuccess          hipSuccess
  #define cudaError_t          hipError_t
  /* HIP builds must use dmabuf path */
  #define TEST_BUF_FLAGS  UGDS_REGISTER_DMABUF
#else
  #include <cuda_runtime.h>
  #define TEST_BUF_FLAGS  0
#endif

#include "bench_utils.h"
#include "ugds.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                       \
    } while (0)

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device> [gpu_id] [seg_size] [nr_segs]\n",
                argv[0]);
        return 1;
    }
    const char* dev = argv[1];
    int gpu_id = (argc > 2) ? atoi(argv[2]) : 0;
    size_t seg_size = (argc > 3) ? strtoull(argv[3], NULL, 0) : 4096;
    unsigned nr_segs = (argc > 4) ? (unsigned)strtoul(argv[4], NULL, 0) : 64;

    CHECK_CUDA(cudaSetDevice(gpu_id));

    uGDSError_t st = uGDSDriverOpen();
    if (st.err != UGDS_SUCCESS) {
        fprintf(stderr, "DriverOpen: %s\n", uGDS_status_error(st.err));
        return 1;
    }

    int fd = open(dev, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    uGDSDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    uGDSHandle_t fh = nullptr;
    st = uGDSHandleRegister(&fh, &descr);
    if (st.err != UGDS_SUCCESS) {
        fprintf(stderr, "HandleRegister: %s\n", uGDS_status_error(st.err));
        return 1;
    }

    const size_t alloc_size = seg_size * nr_segs;
    void* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, alloc_size));
    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    if (st.err != UGDS_SUCCESS) {
        fprintf(stderr, "BufRegister: %s\n", uGDS_status_error(st.err));
        return 1;
    }

    uGDSIoSegment_t* segs =
        (uGDSIoSegment_t*)malloc(nr_segs * sizeof(*segs));
    for (unsigned i = 0; i < nr_segs; i++) {
        segs[i].base = d_buf;
        segs[i].offset = (off_t)(i * seg_size);
        segs[i].size = seg_size;
    }

    /* Warm up both paths. */
    (void)uGDSWritev(fh, segs, nr_segs, 0);
    for (unsigned i = 0; i < nr_segs; i++)
        (void)uGDSWrite(fh, d_buf, seg_size, (off_t)(i * seg_size),
                        i * seg_size);

    const unsigned reps = 200;
    double t0, t1;

    /* Scalar write: nr_segs commands per rep. */
    t0 = now_sec();
    for (unsigned r = 0; r < reps; r++)
        for (unsigned i = 0; i < nr_segs; i++)
            (void)uGDSWrite(fh, d_buf, seg_size, (off_t)(i * seg_size),
                            i * seg_size);
    t1 = now_sec();
    double scalar_w = (double)reps * alloc_size / (t1 - t0);

    /* Vectored write: 1 command stream per rep. */
    t0 = now_sec();
    for (unsigned r = 0; r < reps; r++)
        (void)uGDSWritev(fh, segs, nr_segs, 0);
    t1 = now_sec();
    double vect_w = (double)reps * alloc_size / (t1 - t0);

    /* Scalar read. */
    t0 = now_sec();
    for (unsigned r = 0; r < reps; r++)
        for (unsigned i = 0; i < nr_segs; i++)
            (void)uGDSRead(fh, d_buf, seg_size, (off_t)(i * seg_size),
                           i * seg_size);
    t1 = now_sec();
    double scalar_r = (double)reps * alloc_size / (t1 - t0);

    /* Vectored read. */
    t0 = now_sec();
    for (unsigned r = 0; r < reps; r++)
        (void)uGDSReadv(fh, segs, nr_segs, 0);
    t1 = now_sec();
    double vect_r = (double)reps * alloc_size / (t1 - t0);

    printf("seg_size=%zu nr_segs=%u total=%zu reps=%u\n",
           seg_size, nr_segs, alloc_size, reps);
    printf("scalar  write: %8.2f MiB/s\n", scalar_w / (1024 * 1024));
    printf("vectored write: %8.2f MiB/s  (%.2fx)\n",
           vect_w / (1024 * 1024), vect_w / scalar_w);
    printf("scalar  read : %8.2f MiB/s\n", scalar_r / (1024 * 1024));
    printf("vectored read : %8.2f MiB/s  (%.2fx)\n",
           vect_r / (1024 * 1024), vect_r / scalar_r);

    free(segs);
    uGDSBufDeregister(d_buf);
    CHECK_CUDA(cudaFree(d_buf));
    uGDSHandleDeregister(fh);
    close(fd);
    uGDSDriverClose();
    return 0;
}
