// Functional test for NVMe-oF remote handles.
// - Always runs: invalid-fd error path (no HW needed).
// - With NVMEOF_TEST_DEV: read-only checks (register, uGDSRead vs pread,
//   1 MiB read, 16-entry batch, uGDSReadAsync).
// - NVMEOF_TEST_GPU_DIRECT=1 registers with UGDS_NVMEOF_GPU_DIRECT.
// - Writes need NVMEOF_TEST_ALLOW_WRITE=1 AND an unmounted device; they go to
//   16 GiB, and mounted devices are refused.
// Exit 77 = SKIP (matches scripts/run_tests.sh convention).
// CUDA-only: remote handles need the CUDA backend (see CMakeLists.txt).
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>

#include "ugds.h"

// Host function that holds its stream until *gate becomes non-zero.
static void CUDART_CB wait_gate(void *gate) {
  while (!static_cast<std::atomic<int> *>(gate)->load()) sched_yield();
}

#define CHECK_CUDA(x)                                                    \
  do {                                                                   \
    cudaError_t _e = (x);                                                \
    if (_e != cudaSuccess) {                                             \
      printf("SKIP: cuda unavailable (%s)\n", cudaGetErrorString(_e));   \
      return 77;                                                         \
    }                                                                    \
  } while (0)

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  // --- 1. Invalid fd must fail cleanly (no HW needed) ---
  {
    uGDSError_t e = uGDSDriverOpen();
    if (e.err != UGDS_SUCCESS && e.err != UGDS_DRIVER_ALREADY_OPEN) {
      printf("SKIP: driver open unavailable\n");
      return 77;
    }
    uGDSHandle_t fh = nullptr;
    e = uGDSHandleRegisterNvmeof(&fh, -1);
    if (e.err == UGDS_SUCCESS) {
      printf("FAIL: register(-1) unexpectedly succeeded\n");
      return 1;
    }
    printf("PASS: register(-1) rejected (%d)\n", (int)e.err);
    uGDSDriverClose();
  }

  // --- 2. Full roundtrip needs a real NVMe-oF device ---
  // SAFETY: offset-0 WRITES destroy partition tables/superblocks.
  // Read-only checks run by default; writes require NVMEOF_TEST_ALLOW_WRITE=1
  // AND a scratch device (never a mounted disk — the test refuses those).
  const char *dev = getenv("NVMEOF_TEST_DEV");
  if (!dev || !dev[0]) {
    printf("SKIP: set NVMEOF_TEST_DEV=/dev/nvmeXnY for roundtrip\n");
    return 77;
  }

  int fd = open(dev, O_RDWR | O_DIRECT);
  if (fd < 0) {
    perror("open");
    printf("SKIP: cannot open %s\n", dev);
    return 77;
  }

  uGDSError_t e = uGDSDriverOpen();
  if (e.err != UGDS_SUCCESS && e.err != UGDS_DRIVER_ALREADY_OPEN) {
    printf("SKIP: driver open failed\n");
    close(fd);
    return 77;
  }

  // NVMEOF_TEST_GPU_DIRECT=1 runs every check below over the GPU-direct
  // initiator instead of the host bounce.
  const char *direct = getenv("NVMEOF_TEST_GPU_DIRECT");
  unsigned flags = direct && direct[0] == '1' ? UGDS_NVMEOF_GPU_DIRECT : 0;
  uGDSHandle_t fh = nullptr;
  e = uGDSHandleRegisterNvmeofEx(&fh, fd, flags);
  if (flags && e.err != UGDS_SUCCESS) {
    printf("FAIL: GPU-direct register err=%d\n", (int)e.err);
    close(fd);
    return 1;
  }
  if (flags) printf("PASS: registered with UGDS_NVMEOF_GPU_DIRECT\n");
  if (e.err != UGDS_SUCCESS) {
    printf("SKIP: not an NVMe-oF block device? err=%d\n", (int)e.err);
    close(fd);
    return 77;
  }

  const size_t kSize = 4096;
  void *gpu = nullptr;
  CHECK_CUDA(cudaMalloc(&gpu, kSize));
  char *host_w = nullptr, *host_r = nullptr;
  if (posix_memalign((void **)&host_w, 4096, kSize) ||
      posix_memalign((void **)&host_r, 4096, kSize)) {
    printf("FAIL: host alloc\n");
    return 1;
  }

  // Refuse to WRITE to a mounted filesystem (would destroy data).
  // Default: read-only roundtrip (pread vs uGDSRead comparison).
  // Full write roundtrip only with NVMEOF_TEST_ALLOW_WRITE=1 on a scratch dev.
  const char *allow_write = getenv("NVMEOF_TEST_ALLOW_WRITE");
  bool do_write = allow_write && allow_write[0] == '1';
  if (do_write) {
    FILE *mf = fopen("/proc/mounts", "r");
    char line[512];
    bool mounted = false;
    while (mf && fgets(line, sizeof(line), mf)) {
      if (strstr(line, dev)) { mounted = true; break; }
    }
    if (mf) fclose(mf);
    if (mounted) {
      printf("REFUSE: %s is mounted; unset NVMEOF_TEST_ALLOW_WRITE or use scratch dev\n", dev);
      return 1;
    }
  }
  for (size_t i = 0; i < kSize; ++i) host_w[i] = (char)((i * 131 + 7) & 0xff);
  memset(host_r, 0, kSize);

  // 16 GiB in: well past partition tables and filesystem superblocks.
  const off_t kOff = 16LL << 30;
  if (do_write) {
    CHECK_CUDA(cudaMemcpy(gpu, host_w, kSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());  // GPU-direct: NIC reads outside CUDA order
    if (uGDSWrite(fh, gpu, kSize, kOff, 0) != (ssize_t)kSize) {
      printf("FAIL: uGDSWrite\n");
      return 1;
    }
    printf("PASS: uGDSWrite 4K OK\n");
  }
  // Read path (always safe): compare uGDSRead against kernel pread.
  // GPU-direct: the target RDMA-writes outside CUDA stream order, so the
  // clearing memset must finish first or it can land on top of the data.
  CHECK_CUDA(cudaMemset(gpu, 0, kSize));
  CHECK_CUDA(cudaDeviceSynchronize());
  if (uGDSRead(fh, gpu, kSize, kOff, 0) != (ssize_t)kSize) {
    printf("FAIL: uGDSRead\n");
    return 1;
  }
  CHECK_CUDA(cudaMemcpy(host_r, gpu, kSize, cudaMemcpyDeviceToHost));
  char *host_k = nullptr;
  if (posix_memalign((void **)&host_k, 4096, kSize) ||
      pread(fd, host_k, kSize, kOff) != (ssize_t)kSize) {
    printf("FAIL: kernel pread\n");
    return 1;
  }
  if (memcmp(host_k, host_r, kSize) != 0) {
    printf("FAIL: uGDSRead mismatch vs kernel pread\n");
    return 1;
  }
  if (do_write && memcmp(host_w, host_r, kSize) != 0) {
    printf("FAIL: read-back differs from what uGDSWrite wrote\n");
    return 1;
  }
  printf("PASS: uGDSRead matches kernel pread (4K @16GiB)\n");

  // Pipeline path: one read larger than the 128K chunk.
  {
    const size_t big = 1 << 20;
    void *gpu_big = nullptr;
    CHECK_CUDA(cudaMalloc(&gpu_big, big));
    CHECK_CUDA(cudaMemset(gpu_big, 0, big));
    CHECK_CUDA(cudaDeviceSynchronize());  // see the 4K read above
    if (uGDSRead(fh, gpu_big, big, 1 << 20, 0) != (ssize_t)big) {
      printf("FAIL: uGDSRead 1M\n");
      return 1;
    }
    void *host_big = nullptr;
    void *kern_big = nullptr;
    if (posix_memalign(&host_big, 4096, big) || posix_memalign(&kern_big, 4096, big)) {
      printf("FAIL: 1M host alloc\n");
      return 1;
    }
    CHECK_CUDA(cudaMemcpy(host_big, gpu_big, big, cudaMemcpyDeviceToHost));
    if (pread(fd, kern_big, big, 1 << 20) != (ssize_t)big) {
      printf("FAIL: kernel pread 1M\n");
      return 1;
    }
    if (memcmp(host_big, kern_big, big) != 0) {
      printf("FAIL: 1M uGDSRead mismatch vs kernel pread\n");
      return 1;
    }
    printf("PASS: uGDSRead 1M matches kernel pread\n");
    free(host_big);
    free(kern_big);
    cudaFree(gpu_big);
  }

  // Batch submits many reads and must come back byte-identical.
  {
    const unsigned nio = 16;
    const size_t one = 4096;
    void *gpu_b = nullptr;
    CHECK_CUDA(cudaMalloc(&gpu_b, nio * one));
    CHECK_CUDA(cudaMemset(gpu_b, 0, nio * one));
    CHECK_CUDA(cudaDeviceSynchronize());  // see the 4K read above
    uGDSIOParams_t ps[16];
    memset(ps, 0, sizeof(ps));
    for (unsigned i = 0; i < nio; ++i) {
      ps[i].devPtr_base = gpu_b;
      ps[i].file_offset = (off_t)(i * one);
      ps[i].devPtr_offset = (off_t)(i * one);
      ps[i].size = one;
      ps[i].opcode = UGDS_READ;
      ps[i].cookie = (void *)(uintptr_t)i;
    }
    uGDSBatchHandle_t batch = nullptr;
    e = uGDSBatchIOSetUp(&batch, fh, nio);
    if (e.err != UGDS_SUCCESS) {
      printf("FAIL: remote batch setup %d\n", (int)e.err);
      return 1;
    }
    e = uGDSBatchIOSubmit(batch, nio, ps, 0);
    if (e.err != UGDS_SUCCESS) {
      printf("FAIL: remote batch submit %d\n", (int)e.err);
      return 1;
    }
    uGDSIOEvents_t ev[16];
    unsigned nr = nio;
    struct timespec ts = {5, 0};
    e = uGDSBatchIOGetStatus(batch, nio, &nr, ev, &ts);
    if (e.err != UGDS_SUCCESS || nr != nio) {
      printf("FAIL: remote batch status err=%d nr=%u\n", (int)e.err, nr);
      return 1;
    }
    char *got = (char *)malloc(nio * one);
    CHECK_CUDA(cudaMemcpy(got, gpu_b, nio * one, cudaMemcpyDeviceToHost));
    // fd is O_DIRECT: the reference buffer must be block aligned.
    char *kern = nullptr;
    if (posix_memalign((void **)&kern, 4096, one)) {
      printf("FAIL: batch reference alloc\n");
      return 1;
    }
    for (unsigned i = 0; i < nr; ++i) {
      unsigned idx = (unsigned)(uintptr_t)ev[i].cookie;
      if (ev[i].status != UGDS_BATCH_COMPLETE || ev[i].ret != (ssize_t)one ||
          idx >= nio) {
        printf("FAIL: batch event %u status=%d ret=%ld\n", i, (int)ev[i].status,
               (long)ev[i].ret);
        return 1;
      }
      ssize_t n = pread(fd, kern, one, (off_t)(idx * one));
      if (n != (ssize_t)one) {
        printf("FAIL: batch reference pread at %u returned %ld (%s)\n", idx, (long)n,
               n < 0 ? strerror(errno) : "short read");
        return 1;
      }
      if (memcmp(kern, got + idx * one, one) != 0) {
        printf("FAIL: batch read mismatch at %u\n", idx);
        return 1;
      }
    }
    printf("PASS: remote batch 16x4K matches kernel pread\n");
    free(kern);
    free(got);
    uGDSBatchIODestroy(batch);
    cudaFree(gpu_b);
  }

  // Async on a remote handle must not need uGDSBufRegister (which needs a
  // local controller to map the buffer).
  {
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));
    CHECK_CUDA(cudaMemset(gpu, 0, kSize));
    size_t sz = kSize;
    off_t foff = kOff, boff = 0;
    ssize_t got = 0;
    e = uGDSReadAsync(fh, gpu, &sz, &foff, &boff, &got, st);
    if (e.err != UGDS_SUCCESS) {
      printf("FAIL: remote uGDSReadAsync %d\n", (int)e.err);
      return 1;
    }
    CHECK_CUDA(cudaStreamSynchronize(st));
    CHECK_CUDA(cudaMemcpy(host_r, gpu, kSize, cudaMemcpyDeviceToHost));
    if (got != (ssize_t)kSize || memcmp(host_k, host_r, kSize) != 0) {
      printf("FAIL: remote uGDSReadAsync ret=%ld or data mismatch\n", (long)got);
      return 1;
    }
    cudaStreamDestroy(st);
    printf("PASS: remote uGDSReadAsync matches kernel pread\n");
  }

  // A remote async request queued with an unregistered buffer holds no
  // buffer reference. If the buffer is registered before the callback runs,
  // the callback must not drop a reference it never took, or the buffer
  // can never be deregistered (UGDS_BUSY). A host function holds the stream
  // until the registration is done. uGDSBufRegister needs a local
  // controller; without one this check is skipped.
  {
    const size_t len = 2 << 20;
    void *gpu_a = nullptr;
    CHECK_CUDA(cudaMalloc(&gpu_a, len));
    cudaStream_t st;
    // Non-blocking, so a legacy-stream call inside uGDSBufRegister cannot
    // wait on the gated stream and hang the test.
    CHECK_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    std::atomic<int> gate{0};
    CHECK_CUDA(cudaLaunchHostFunc(st, wait_gate, &gate));
    size_t sz = kSize;
    off_t foff = kOff, boff = 0;
    ssize_t got = 0;
    e = uGDSReadAsync(fh, gpu_a, &sz, &foff, &boff, &got, st);
    uGDSError_t re = uGDSBufRegister(gpu_a, len, 0);
    gate.store(1);
    CHECK_CUDA(cudaStreamSynchronize(st));
    cudaStreamDestroy(st);
    if (e.err != UGDS_SUCCESS || got != (ssize_t)kSize) {
      printf("FAIL: gated remote uGDSReadAsync err=%d ret=%ld\n", (int)e.err, (long)got);
      return 1;
    }
    if (re.err != UGDS_SUCCESS) {
      printf("SKIP: register during async (uGDSBufRegister err=%d, no local controller?)\n",
             (int)re.err);
    } else {
      uGDSError_t de = uGDSBufDeregister(gpu_a);
      if (de.err != UGDS_SUCCESS) {
        printf("FAIL: deregister after async returned %d (reference count broken)\n",
               (int)de.err);
        return 1;
      }
      printf("PASS: register during remote async keeps the reference count\n");
    }
    cudaFree(gpu_a);
  }

  uGDSHandleDeregister(fh);  // fd ownership stays with caller
  close(fd);
  cudaFree(gpu);
  free(host_w);
  free(host_r);
  free(host_k);
  uGDSDriverClose();
  printf("PASS: test_nvmeof_register\n");
  return 0;
}
