// Read bench for a uGDS NVMe-oF handle.
// Sync path: one command in flight per thread.
// Batch path: one thread, many commands in flight.
// Reads only, random offsets in 32..40 GiB, which must already hold data.
// UGDS_BENCH_GPU_DIRECT=1 registers with UGDS_NVMEOF_GPU_DIRECT.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ugds.h"

static double now_s() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static uGDSHandle_t g_fh;
static size_t g_sz;
static int g_iters;
static int g_base_cpu = 32;
static pthread_barrier_t g_bar;
// Offsets differ on every run unless UGDS_BENCH_SEED is set: the target
// caches recently read blocks, so a repeated sequence of offsets measures
// that cache instead of the disk.
static unsigned g_seed;

struct SyncArg {
  int id;
  int rc;
};

static void *sync_worker(void *arg) {
  SyncArg *a = (SyncArg *)arg;
  pin(g_base_cpu + (a->id % 16));
  // 32-40 GiB is fully written (see paper-tests prep). Unwritten LBAs read
  // back in ~20 us and recently read LBAs hit the target's cache, so random
  // offsets in a written region are the only honest latency.
  off_t base = (32LL << 30) + (off_t)(a->id % 8) * (1LL << 30);
  void *gpu = nullptr;
  if (cudaMalloc(&gpu, g_sz) != cudaSuccess) {
    a->rc = 1;
    pthread_barrier_wait(&g_bar);
    return nullptr;
  }
  if (uGDSRead(g_fh, gpu, g_sz, base, 0) != (ssize_t)g_sz) {
    a->rc = 2;
    cudaFree(gpu);
    pthread_barrier_wait(&g_bar);
    return nullptr;
  }
  pthread_barrier_wait(&g_bar);
  unsigned seed = g_seed + (unsigned)a->id * 7919u;
  const unsigned long slots = (1UL << 30) / g_sz;
  for (int i = 0; i < g_iters; ++i) {
    off_t off = base + (off_t)((rand_r(&seed) % slots) * g_sz);
    if (uGDSRead(g_fh, gpu, g_sz, off, 0) != (ssize_t)g_sz) {
      a->rc = 3;
      break;
    }
  }
  cudaFree(gpu);
  return nullptr;
}

static int run_sync(int nthreads) {
  pthread_t th[64];
  SyncArg args[64];
  if (nthreads > 64) nthreads = 64;
  pthread_barrier_init(&g_bar, nullptr, (unsigned)nthreads + 1);
  for (int i = 0; i < nthreads; ++i) {
    args[i].id = i;
    args[i].rc = 0;
    pthread_create(&th[i], nullptr, sync_worker, &args[i]);
  }
  pthread_barrier_wait(&g_bar);
  double t0 = now_s();
  int fails = 0;
  for (int i = 0; i < nthreads; ++i) {
    pthread_join(th[i], nullptr);
    if (args[i].rc) fails++;
  }
  double dt = now_s() - t0;
  pthread_barrier_destroy(&g_bar);
  double bytes = (double)g_sz * (double)g_iters * (double)nthreads;
  double gib = bytes / (1 << 30) / dt;
  double iops = bytes / (double)g_sz / dt;
  printf("sync  threads=%2d size=%7zu  %7.1f K IOPS   %5.2f GiB/s  fails=%d\n",
         nthreads, g_sz, iops / 1e3, gib, fails);
  return fails ? 1 : 0;
}

static int run_batch(unsigned nio, int rounds) {
  pin(g_base_cpu);
  void *gpu = nullptr;
  if (cudaMalloc(&gpu, (size_t)nio * g_sz) != cudaSuccess) return 1;
  uGDSIOParams_t *ps =
      (uGDSIOParams_t *)calloc(nio, sizeof(uGDSIOParams_t));
  uGDSIOEvents_t *ev =
      (uGDSIOEvents_t *)calloc(nio, sizeof(uGDSIOEvents_t));
  if (!ps || !ev) return 1;
  uGDSBatchHandle_t batch = nullptr;
  uGDSError_t e = uGDSBatchIOSetUp(&batch, g_fh, nio);
  if (e.err != UGDS_SUCCESS) {
    printf("batch setup failed %d\n", (int)e.err);
    return 1;
  }
  off_t base = 32LL << 30;  // written region, see sync_worker
  unsigned seed = g_seed ^ 0x9e3779b9u;
  const unsigned long slots = (8UL << 30) / g_sz;
  // Warm the workers and the CUDA context.
  for (unsigned i = 0; i < nio; ++i) {
    ps[i].devPtr_base = gpu;
    ps[i].devPtr_offset = (off_t)i * (off_t)g_sz;
    ps[i].size = g_sz;
    ps[i].opcode = UGDS_READ;
    ps[i].file_offset = base + ps[i].devPtr_offset;
  }
  if (uGDSBatchIOSubmit(batch, nio, ps, 0).err != UGDS_SUCCESS) return 1;
  unsigned nr = nio;
  struct timespec ts = {10, 0};
  if (uGDSBatchIOGetStatus(batch, nio, &nr, ev, &ts).err != UGDS_SUCCESS ||
      nr != nio)
    return 1;

  double t0 = now_s();
  int fails = 0;
  for (int r = 0; r < rounds; ++r) {
    for (unsigned i = 0; i < nio; ++i) {
      // 4K: random like fio randread. Larger: sequential, wrapping in 8 GiB.
      unsigned long slot = g_sz <= 4096
          ? rand_r(&seed) % slots
          : ((unsigned long)r * nio + i) % slots;
      ps[i].file_offset = base + (off_t)(slot * g_sz);
    }
    if (uGDSBatchIOSubmit(batch, nio, ps, 0).err != UGDS_SUCCESS) {
      fails++;
      break;
    }
    nr = nio;
    if (uGDSBatchIOGetStatus(batch, nio, &nr, ev, &ts).err != UGDS_SUCCESS ||
        nr != nio) {
      fails++;
      break;
    }
    for (unsigned i = 0; i < nr; ++i) {
      if (ev[i].status != UGDS_BATCH_COMPLETE || ev[i].ret != (ssize_t)g_sz)
        fails++;
    }
    if (fails) break;
  }
  double dt = now_s() - t0;
  double bytes = (double)g_sz * (double)nio * (double)rounds;
  double gib = bytes / (1 << 30) / dt;
  double iops = bytes / (double)g_sz / dt;
  printf("batch nio=%3u rounds=%3d size=%7zu  %7.1f K IOPS   %5.2f GiB/s  fails=%d\n",
         nio, rounds, g_sz, iops / 1e3, gib, fails);
  uGDSBatchIODestroy(batch);
  free(ps);
  free(ev);
  cudaFree(gpu);
  return fails ? 1 : 0;
}

int main(int argc, char **argv) {
  const char *dev = argc > 1 ? argv[1] : "/dev/nvme1n1";
  int fd = open(dev, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    perror("open");
    return 1;
  }
  if (uGDSDriverOpen().err != UGDS_SUCCESS) return 77;
  const char *direct = getenv("UGDS_BENCH_GPU_DIRECT");
  unsigned flags = direct && direct[0] == '1' ? UGDS_NVMEOF_GPU_DIRECT : 0;
  if (uGDSHandleRegisterNvmeofEx(&g_fh, fd, flags).err != UGDS_SUCCESS) return 1;
  const char *seed_env = getenv("UGDS_BENCH_SEED");
  g_seed = seed_env ? (unsigned)strtoul(seed_env, nullptr, 0)
                    : (unsigned)time(nullptr) ^ ((unsigned)getpid() << 16);
  printf("mode: %s, seed %u\n", flags ? "GPU-direct (no host copy)" : "host bounce", g_seed);

  printf("baseline was sync 4K x16 = 108K IOPS, sync 128K x8 = 1.33 GiB/s\n");
  printf("target is about 300K IOPS and 3 GiB/s read on this single disk\n");

  g_sz = 4096;
  g_iters = 2000;
  int rc = 0;
  for (int t = 1; t <= 32; t *= 2) rc |= run_sync(t);

  g_sz = 131072;
  g_iters = 400;
  for (int t = 1; t <= 32; t *= 2) rc |= run_sync(t);

  g_sz = 4096;
  rc |= run_batch(64, 800);
  rc |= run_batch(128, 800);

  g_sz = 131072;
  rc |= run_batch(64, 120);
  rc |= run_batch(128, 80);

  uGDSHandleDeregister(g_fh);
  uGDSDriverClose();
  close(fd);
  return rc ? 1 : 0;
}
