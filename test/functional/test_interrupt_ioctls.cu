// Phase 6 Stage 1 — 中断模式内核基础设施的行为测试
//
// 验证内核驱动新增的三个 ioctl 的最小契约：
//   NVM_GET_NUM_VECTORS       查询已分配的 MSI-X 向量数（应 > 0）
//   NVM_REGISTER_INTERRUPT    把一个 eventfd 绑定到某个向量
//   NVM_UNREGISTER_INTERRUPT  解绑
//
// 本测试不涉及 GPU/CUDA，只直接对设备 fd 发 ioctl。
//
// RED：在内核实现这些 ioctl 之前，NVM_GET_NUM_VECTORS 会返回 -1/EINVAL
//      （未知 ioctl 命令），测试失败。
// GREEN：内核实现后，全部契约满足，测试通过。

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

// ── 期望的 ioctl ABI 契约（阶段1将在 drv/ioctl.h + libnvm/internal/ioctl.h 实现）──
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

    // 1. NVM_GET_NUM_VECTORS 应返回 >0 的向量数
    __u32 num_vectors = 0;
    if (ioctl(fd, NVM_GET_NUM_VECTORS, &num_vectors) != 0)
        FAIL("NVM_GET_NUM_VECTORS: %s", strerror(errno));
    if (num_vectors == 0)
        FAIL("NVM_GET_NUM_VECTORS returned 0 vectors (MSI-X not allocated)");
    printf("  num_vectors = %u\n", num_vectors);

    // 2. 用 eventfd 注册向量 0 应成功
    int efd = eventfd(0, EFD_CLOEXEC);
    if (efd < 0) FAIL("eventfd: %s", strerror(errno));

    struct nvm_ioctl_irq reg;
    memset(&reg, 0, sizeof(reg));
    reg.vector = 0;
    reg.eventfd = efd;
    if (ioctl(fd, NVM_REGISTER_INTERRUPT, &reg) != 0)
        FAIL("NVM_REGISTER_INTERRUPT(vector=0): %s", strerror(errno));

    // 3. 重复注册同一向量应返回 EBUSY
    int rc = ioctl(fd, NVM_REGISTER_INTERRUPT, &reg);
    if (rc == 0)
        FAIL("duplicate NVM_REGISTER_INTERRUPT should fail with EBUSY, but succeeded");
    if (errno != EBUSY)
        FAIL("duplicate register: expected EBUSY, got %s", strerror(errno));

    // 4. 注销向量 0 应成功
    if (ioctl(fd, NVM_UNREGISTER_INTERRUPT, &reg) != 0)
        FAIL("NVM_UNREGISTER_INTERRUPT(vector=0): %s", strerror(errno));

    // 5. 越界向量应返回 EINVAL
    struct nvm_ioctl_irq bad;
    memset(&bad, 0, sizeof(bad));
    bad.vector = num_vectors;   // 越界（合法范围 0..num_vectors-1）
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
