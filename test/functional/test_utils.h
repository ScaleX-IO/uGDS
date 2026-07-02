#ifndef __UGDS_TEST_UTILS_H__
#define __UGDS_TEST_UTILS_H__

#include <ugds.h>

#ifdef __HIP_PLATFORM_AMD__
  #include <hip/hip_runtime.h>
  #define cudaMalloc            hipMalloc
  #define cudaFree              hipFree
  #define cudaMemcpy           hipMemcpy
  #define cudaMemset            hipMemset
  #define cudaSetDevice         hipSetDevice
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaSuccess           hipSuccess
  #define cudaError_t           hipError_t
  #define cudaGetErrorString    hipGetErrorString
  #define cudaStreamSynchronize hipStreamSynchronize
  #define cudaStreamCreate      hipStreamCreate
  #define cudaStreamDestroy     hipStreamDestroy
  #define cudaMemsetAsync       hipMemsetAsync
  #define cudaHostAlloc         hipHostMalloc
  #define cudaHostAllocDefault  hipHostMallocDefault
  #define cudaFreeHost          hipHostFree
  #define cudaLaunchHostFunc    hipLaunchHostFunc
  #define cudaStream_t          hipStream_t
  /* HIP builds must use dmabuf path */
  #define TEST_BUF_FLAGS  UGDS_REGISTER_DMABUF
#else
  #include <cuda_runtime.h>
  #define TEST_BUF_FLAGS  0
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static const char* g_dev_path = nullptr;
static int g_gpu_id = 0;

#define TEST_PASS() \
    do { printf("PASS\n"); return 0; } while(0)
#define TEST_FAIL(fmt, ...) \
    do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)
#define ASSERT_OK(st, msg) \
    do { if ((st).err != UGDS_SUCCESS) TEST_FAIL("%s: %s", msg, uGDS_status_error((st).err)); } while(0)
#define ASSERT_ERR(st, expected, msg) \
    do { if ((st).err != (expected)) TEST_FAIL("%s: expected %s, got %s", msg, \
        uGDS_status_error(expected), uGDS_status_error((st).err)); } while(0)

__global__ void fill_pattern_u32(uint32_t* buf, uint32_t pattern, size_t n_words) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_words) buf[idx] = pattern;
}

static uGDSHandle_t open_handle() {
    int fd = open(g_dev_path, O_RDWR);
    if (fd < 0) return nullptr;
    uGDSDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    uGDSHandle_t fh = nullptr;
    uGDSError_t st = uGDSHandleRegister(&fh, &descr);
    if (st.err != UGDS_SUCCESS) {
        close(fd);
        return nullptr;
    }
    return fh;
}

static int get_handle_fd(uGDSHandle_t fh) {
    if (!fh) return -1;
    return *reinterpret_cast<int*>(fh);
}

static void close_handle(uGDSHandle_t fh) {
    if (!fh) return;
    int fd = get_handle_fd(fh);
    uGDSHandleDeregister(fh);
    if (fd >= 0) close(fd);
}

static bool parse_args(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device_path> [gpu_id]\n", argv[0]);
        return false;
    }
    g_dev_path = argv[1];
    g_gpu_id = (argc > 2) ? atoi(argv[2]) : 0;
    return true;
}

#endif
