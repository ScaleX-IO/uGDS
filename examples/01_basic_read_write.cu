// Basic uGDS read/write example
// Build: nvcc -o basic_rw 01_basic_read_write.cu -lugds -lcudart
// Run:   ./basic_rw /dev/ugds_drv0

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ugds.h>
#include <fcntl.h>
#include <unistd.h>

#define CHECK_UGDS(call)                                          \
    do {                                                            \
        uGDSError_t err = (call);                                 \
        if (err.err != UGDS_SUCCESS) {                           \
            fprintf(stderr, "uGDS error: %s\n",                      \
                    uGDS_status_error(err.err));                 \
            exit(1);                                                \
        }                                                           \
    } while (0)

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error: %s\n",                     \
                    cudaGetErrorString(err));                        \
            exit(1);                                                \
        }                                                           \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device_path> [gpu_id]\n", argv[0]);
        return 1;
    }

    const char* dev_path = argv[1];
    int gpu_id = (argc > 2) ? atoi(argv[2]) : 0;
    const size_t IO_SIZE = 4096;

    CHECK_CUDA(cudaSetDevice(gpu_id));

    // 1. Initialize driver
    CHECK_UGDS(uGDSDriverOpen());

    // 2. Open device and register handle
    int fd = open(dev_path, O_RDWR);
    uGDSDescr_t desc = {};
    desc.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = fd;

    uGDSHandle_t fh;
    CHECK_UGDS(uGDSHandleRegister(&fh, &desc));

    // 3. Allocate GPU buffer and register with uGDS
    void* gpu_buf;
    CHECK_CUDA(cudaMalloc(&gpu_buf, IO_SIZE));
    CHECK_UGDS(uGDSBufRegister(gpu_buf, IO_SIZE, 0));

    // 4. Fill GPU buffer with a pattern and write to SSD
    CHECK_CUDA(cudaMemset(gpu_buf, 0xAB, IO_SIZE));
    ssize_t written = uGDSWrite(fh, gpu_buf, IO_SIZE, 0, 0);
    printf("Written: %zd bytes\n", written);

    // 5. Clear GPU buffer, then read back from SSD
    CHECK_CUDA(cudaMemset(gpu_buf, 0x00, IO_SIZE));
    ssize_t read_bytes = uGDSRead(fh, gpu_buf, IO_SIZE, 0, 0);
    printf("Read:    %zd bytes\n", read_bytes);

    // 6. Verify on host
    unsigned char* host_buf = (unsigned char*)malloc(IO_SIZE);
    CHECK_CUDA(cudaMemcpy(host_buf, gpu_buf, IO_SIZE, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < IO_SIZE; i++) {
        if (host_buf[i] != 0xAB) { ok = false; break; }
    }
    printf("Verify:  %s\n", ok ? "PASS" : "FAIL");

    // 7. Cleanup
    free(host_buf);
    uGDSBufDeregister(gpu_buf);
    CHECK_CUDA(cudaFree(gpu_buf));
    uGDSHandleDeregister(fh);
    close(fd);
    uGDSDriverClose();

    return ok ? 0 : 1;
}
