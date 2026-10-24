// GPU-direct NVMe-oF check: the capsule initiator reads into GPU memory and
// the result must match what the kernel nvme-rdma device returns.
//
//   NVMEOF_TEST_DEV=/dev/nvme1n1 ./test_nvmeof_direct
//
// Reads come from NVMEOF_TEST_READ_GIB (default 32) .. +8 GiB, which must
// already hold data. Writes run only with NVMEOF_TEST_ALLOW_WRITE=1, at
// NVMEOF_TEST_WRITE_GIB (default 16, minimum 16), on an unmounted device.
// Exit 77 = SKIP.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#include "../../src/ugds_nvmeof_capsule.h"

static double now_us() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static bool mounted(const char* dev) {
    FILE* f = fopen("/proc/mounts", "r");
    if (!f) return true;
    char line[512];
    bool hit = false;
    size_t n = strlen(dev);
    while (fgets(line, sizeof(line), f))
        if (!strncmp(line, dev, n) && line[n] == ' ') hit = true;
    fclose(f);
    return hit;
}

static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t)(p / 100.0 * (v.size() - 1))];
}

static int compare(const char* what, ugds_capsule_ctrl* c, int fd, void* gpu, void* host,
                   void* ref, size_t n, off_t off) {
    ssize_t r = ugds_capsule_io(c, gpu, n, off, false);
    if (r != (ssize_t)n) {
        printf("FAIL: %s capsule read returned %zd\n", what, r);
        return 1;
    }
    cudaMemcpy(host, gpu, n, cudaMemcpyDeviceToHost);
    if (pread(fd, ref, n, off) != (ssize_t)n) {
        perror("pread");
        return 1;
    }
    if (memcmp(host, ref, n)) {
        printf("FAIL: %s differs from kernel pread at %lld\n", what, (long long)off);
        return 1;
    }
    printf("PASS: %s at %lld GiB matches pread\n", what, (long long)(off >> 30));
    return 0;
}

int main() {
    const char* dev = getenv("NVMEOF_TEST_DEV");
    if (!dev) {
        printf("SKIP: set NVMEOF_TEST_DEV=/dev/nvmeXnY\n");
        return 77;
    }
    bool allow_write = getenv("NVMEOF_TEST_ALLOW_WRITE") && !strcmp(getenv("NVMEOF_TEST_ALLOW_WRITE"), "1");
    int fd = open(dev, (allow_write ? O_RDWR : O_RDONLY) | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 77;
    }

    ugds_capsule_target t;
    int r = ugds_capsule_target_from_fd(fd, &t);
    if (r) {
        printf("SKIP: %s is not a kernel nvme-rdma namespace (%d)\n", dev, r);
        return 77;
    }
    if (getenv("NVMEOF_TEST_QUEUES")) t.nr_io_queues = atoi(getenv("NVMEOF_TEST_QUEUES"));
    printf("target %s:%s nsid %u subnqn %s src %s\n", t.traddr.c_str(), t.trsvcid.c_str(), t.nsid,
           t.subnqn.c_str(), t.host_traddr.empty() ? "(route)" : t.host_traddr.c_str());

    ugds_capsule_ctrl* c = nullptr;
    double t0 = now_us();
    r = ugds_capsule_connect(t, &c);
    if (r) {
        printf("FAIL: connect %d\n", r);
        return 1;
    }
    printf("PASS: connected %d I/O queues in %.0f ms\n", t.nr_io_queues, (now_us() - t0) / 1000);

    unsigned long long kbytes = 0;
    ioctl(fd, BLKGETSIZE64, &kbytes);
    if (ugds_capsule_capacity(c) != kbytes) {
        printf("FAIL: capacity %llu vs kernel %llu\n", (unsigned long long)ugds_capsule_capacity(c), kbytes);
        return 1;
    }
    printf("PASS: capacity %llu bytes, block %u\n", kbytes, ugds_capsule_block_size(c));

    const size_t kBuf = 64u << 20;
    void *gpu = nullptr, *host = nullptr, *ref = nullptr;
    if (cudaMalloc(&gpu, kBuf) != cudaSuccess || posix_memalign(&host, 4096, kBuf) ||
        posix_memalign(&ref, 4096, kBuf)) {
        printf("FAIL: alloc\n");
        return 1;
    }
    cudaMemset(gpu, 0xEE, kBuf);
    cudaDeviceSynchronize();
    t0 = now_us();
    r = ugds_capsule_register(c, gpu, kBuf);
    printf("%s: register 64 MiB GPU buffer (%d) in %.1f ms\n", r ? "FAIL" : "PASS", r,
           (now_us() - t0) / 1000);
    if (r) return 1;

    const off_t base = (off_t)(getenv("NVMEOF_TEST_READ_GIB") ? atoi(getenv("NVMEOF_TEST_READ_GIB")) : 32) << 30;
    const off_t span = 8ll << 30;
    int fails = 0;
    fails += compare("4 KiB read", c, fd, gpu, host, ref, 4096, base);
    fails += compare("128 KiB read", c, fd, gpu, host, ref, 128 << 10, base + (1 << 20));
    fails += compare("1 MiB read", c, fd, gpu, host, ref, 1 << 20, base + (8 << 20));
    fails += compare("32 MiB read", c, fd, gpu, host, ref, 32u << 20, base + (64 << 20));

    // QD1 4 KiB random reads: capsule into GPU vs kernel pread into host.
    const int kIters = 20000;
    std::vector<double> lat_cap, lat_kern;
    unsigned seed = 1;
    for (int i = 0; i < kIters; ++i) {
        off_t o = base + (off_t)(rand_r(&seed) % (span / 4096)) * 4096;
        double a = now_us();
        if (ugds_capsule_io(c, gpu, 4096, o, false) != 4096) { fails++; break; }
        lat_cap.push_back(now_us() - a);
    }
    seed = 2;
    for (int i = 0; i < kIters; ++i) {
        off_t o = base + (off_t)(rand_r(&seed) % (span / 4096)) * 4096;
        double a = now_us();
        if (pread(fd, ref, 4096, o) != 4096) { fails++; break; }
        lat_kern.push_back(now_us() - a);
    }
    printf("4 KiB QD1 random read: GPU-direct p50 %.1f p99 %.1f us | kernel pread p50 %.1f p99 %.1f us\n",
           pct(lat_cap, 50), pct(lat_cap, 99), pct(lat_kern, 50), pct(lat_kern, 99));

    // One thread, sequential 32 MiB reads (32 x 1 MiB commands, depth-limited).
    const int kRounds = 64;
    t0 = now_us();
    for (int i = 0; i < kRounds; ++i)
        if (ugds_capsule_io(c, gpu, 32u << 20, base + (off_t)i * (32 << 20), false) != (32 << 20)) { fails++; break; }
    double cap_s = (now_us() - t0) / 1e6;
    t0 = now_us();
    for (int i = 0; i < kRounds; ++i)
        if (pread(fd, ref, 32u << 20, base + (off_t)i * (32 << 20)) != (32 << 20)) { fails++; break; }
    double kern_s = (now_us() - t0) / 1e6;
    double gib = kRounds * 32.0 / 1024;
    printf("32 MiB sequential read, 1 thread: GPU-direct %.2f GiB/s | kernel pread (host only) %.2f GiB/s\n",
           gib / cap_s, gib / kern_s);

    if (allow_write) {
        off_t woff = (off_t)(getenv("NVMEOF_TEST_WRITE_GIB") ? atoi(getenv("NVMEOF_TEST_WRITE_GIB")) : 16) << 30;
        if (woff < (16ll << 30)) {
            printf("FAIL: refusing write below 16 GiB\n");
            return 1;
        }
        if (mounted(dev)) {
            printf("FAIL: %s is mounted, refusing write\n", dev);
            return 1;
        }
        for (size_t n : {(size_t)4096, (size_t)(1u << 20), (size_t)(16u << 20)}) {
            unsigned char* h = (unsigned char*)host;
            for (size_t i = 0; i < n; ++i) h[i] = (unsigned char)(i * 131 + n + 7);
            // A pageable cudaMemcpy may return before the data lands; the NIC
            // reads GPU memory outside CUDA's ordering.
            cudaMemcpy(gpu, host, n, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            if (ugds_capsule_io(c, gpu, n, woff, true) != (ssize_t)n) {
                printf("FAIL: write %zu\n", n);
                fails++;
                continue;
            }
            if (pread(fd, ref, n, woff) != (ssize_t)n || memcmp(ref, host, n)) {
                printf("FAIL: write %zu not visible to kernel pread\n", n);
                fails++;
            } else {
                printf("PASS: GPU-direct write %zu bytes at %lld GiB, kernel pread matches\n", n,
                       (long long)(woff >> 30));
            }
        }
        std::vector<double> wl;
        for (int i = 0; i < 5000; ++i) {
            double a = now_us();
            if (ugds_capsule_io(c, gpu, 4096, woff + (off_t)(i % 4096) * 4096, true) != 4096) { fails++; break; }
            wl.push_back(now_us() - a);
        }
        printf("4 KiB QD1 write: GPU-direct p50 %.1f p99 %.1f us\n", pct(wl, 50), pct(wl, 99));
        t0 = now_us();
        for (int i = 0; i < 32; ++i)
            if (ugds_capsule_io(c, gpu, 32u << 20, woff + (off_t)i * (32 << 20), true) != (32 << 20)) { fails++; break; }
        printf("32 MiB sequential write, 1 thread: GPU-direct %.2f GiB/s\n", 1.0 / ((now_us() - t0) / 1e6));
    }

    ugds_capsule_disconnect(c);
    cudaFree(gpu);
    free(host);
    free(ref);
    close(fd);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
