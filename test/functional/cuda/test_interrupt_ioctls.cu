// Phase 6 Stage 1 - Behavioral test for the interrupt-mode kernel infrastructure
//
// Verifies the minimal contract of the three new kernel-driver ioctls:
//   NVM_GET_NUM_VECTORS       query the number of allocated MSI-X vectors (> 0)
//   NVM_REGISTER_INTERRUPT    bind an eventfd to a vector
//   NVM_UNREGISTER_INTERRUPT  unbind
//
// This test does not touch GPU/CUDA; it issues ioctls directly on the device fd.
//
// RED: before the kernel implements these ioctls, NVM_GET_NUM_VECTORS returns
//      -1/EINVAL (unknown ioctl command) and the test fails.
// GREEN: once implemented, the whole contract holds and the test passes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/eventfd.h>
#include <linux/types.h>
#include <asm/ioctl.h>

// Expected ioctl ABI contract (Stage 1 implements it in drv/ioctl.h +
// libnvm/internal/ioctl.h).
#define NVM_IOCTL_TYPE 0x80

struct nvm_ioctl_irq {
    __u32 vector;
    __s32 eventfd;
};

#ifndef NVM_REGISTER_INTERRUPT
#define NVM_REGISTER_INTERRUPT   _IOW(NVM_IOCTL_TYPE, 5, struct nvm_ioctl_irq)
#define NVM_UNREGISTER_INTERRUPT _IOW(NVM_IOCTL_TYPE, 6, struct nvm_ioctl_irq)
#define NVM_GET_NUM_VECTORS      _IOR(NVM_IOCTL_TYPE, 7, __u32)
#endif

#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device_path> [gpu_id]\n", argv[0]);
        return 1;
    }
    const char* dev_path = argv[1];

    int fd = open(dev_path, O_RDWR);
    if (fd < 0) FAIL("open(%s): %s", dev_path, strerror(errno));

    // 1. NVM_GET_NUM_VECTORS should return a vector count > 0
    __u32 num_vectors = 0;
    if (ioctl(fd, NVM_GET_NUM_VECTORS, &num_vectors) != 0)
        FAIL("NVM_GET_NUM_VECTORS: %s", strerror(errno));
    if (num_vectors == 0)
        FAIL("NVM_GET_NUM_VECTORS returned 0 vectors (MSI-X not allocated)");
    printf("  num_vectors = %u\n", num_vectors);

    // 2. Registering vector 0 with an eventfd should succeed
    int efd = eventfd(0, EFD_CLOEXEC);
    if (efd < 0) FAIL("eventfd: %s", strerror(errno));

    struct nvm_ioctl_irq reg;
    memset(&reg, 0, sizeof(reg));
    reg.vector = 0;
    reg.eventfd = efd;
    if (ioctl(fd, NVM_REGISTER_INTERRUPT, &reg) != 0)
        FAIL("NVM_REGISTER_INTERRUPT(vector=0): %s", strerror(errno));

    // 3. Re-registering the same vector should return EBUSY
    int rc = ioctl(fd, NVM_REGISTER_INTERRUPT, &reg);
    if (rc == 0)
        FAIL("duplicate NVM_REGISTER_INTERRUPT should fail with EBUSY, but succeeded");
    if (errno != EBUSY)
        FAIL("duplicate register: expected EBUSY, got %s", strerror(errno));

    // 4. Unregistering vector 0 should succeed
    if (ioctl(fd, NVM_UNREGISTER_INTERRUPT, &reg) != 0)
        FAIL("NVM_UNREGISTER_INTERRUPT(vector=0): %s", strerror(errno));

    // 5. An out-of-range vector should return EINVAL
    struct nvm_ioctl_irq bad;
    memset(&bad, 0, sizeof(bad));
    bad.vector = num_vectors;   // out of range (valid: 0..num_vectors-1)
    bad.eventfd = efd;
    rc = ioctl(fd, NVM_REGISTER_INTERRUPT, &bad);
    if (rc == 0)
        FAIL("out-of-range vector should fail with EINVAL, but succeeded");
    if (errno != EINVAL)
        FAIL("out-of-range vector: expected EINVAL, got %s", strerror(errno));

    close(efd);
    close(fd);
    printf("PASS\n");
    return 0;
}
