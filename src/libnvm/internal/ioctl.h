#ifndef __NVM_INTERNAL_LINUX_IOCTL_H__
#define __NVM_INTERNAL_LINUX_IOCTL_H__
#ifdef __linux__

#include <linux/types.h>
#include <asm/ioctl.h>

#define NVM_IOCTL_TYPE          0x80



/* Memory map request */
struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
};

/* Always define the dmabuf struct so the ioctl enum can reference it
 * unconditionally. The handler body is only compiled when
 * UGDS_HAVE_DMABUF is set, but the type must exist for the enum. */
struct nvm_ioctl_dmabuf
{
    uint64_t  gpu_ptr;           /* Original GPU VA -- unmap identity */
    int32_t   dmabuf_fd;         /* DMA-buf fd from GPU export */
    uint32_t  __pad;             /* Alignment */
    uint64_t  dmabuf_offset;     /* Offset within dmabuf allocation */
    uint64_t  size;              /* Total buffer size in bytes */
    uint64_t  ioaddrs_capacity;  /* Max entries in ioaddrs buffer */
    uint64_t  ioaddrs;           /* Output: DMA bus addresses (pointer) */
};

/* Bind/unbind an eventfd to an MSI-X interrupt vector (interrupt mode).
 * Must match struct nvm_ioctl_irq in drv/ioctl.h. */
struct nvm_ioctl_irq
{
    __u32  vector;            /* MSI-X vector index (0 .. num_vectors-1) */
    __s32  eventfd;           /* eventfd to signal on IRQ (register only) */
};


/* --- V2 dma-buf map ioctl (mirrors drv/ioctl.h) ------------------- */

#define NVM_DMABUF_MAP_VERSION_1      1u
#define UGDS_DMABUF_REQUIRE_P2P       (1u << 0)

enum nvm_dmabuf_mapping_class {
    NVM_DMABUF_MAPPING_UNKNOWN  = 0,
    NVM_DMABUF_MAPPING_SYSTEM   = 1,
    NVM_DMABUF_MAPPING_PEER_BAR = 2,
};

enum nvm_dmabuf_failure_reason {
    NVM_DMABUF_FAIL_NONE          = 0,
    NVM_DMABUF_FAIL_BAD_FD        = 1,
    NVM_DMABUF_FAIL_NOT_DMABUF    = 2,
    NVM_DMABUF_FAIL_RANGE         = 3,
    NVM_DMABUF_FAIL_PIN           = 4,
    NVM_DMABUF_FAIL_FENCE         = 5,
    NVM_DMABUF_FAIL_MAP           = 6,
    NVM_DMABUF_FAIL_SG_LAYOUT     = 7,
    NVM_DMABUF_FAIL_NOT_PEER_BAR  = 8,
    NVM_DMABUF_FAIL_PIN_LIMIT     = 9,
};

struct nvm_ioctl_dmabuf_v2 {
    __u32  struct_size;
    __u16  version;
    __u16  flags;
    __u64  gpu_ptr;
    __s32  dmabuf_fd;
    __u32  reserved0;
    __u64  dmabuf_offset;
    __u64  size;
    __u64  ioaddrs_capacity;
    __u64  ioaddrs;
    __u32  mapping_class;
    __u32  failure_reason;
    __u16  peer_domain;
    __u8   peer_bus;
    __u8   peer_devfn;
    __u32  peer_bar;
    __u64  peer_bar_start;
    __u64  peer_bar_length;
    __u64  reserved[4];
};

/* --- Capability query ioctl (mirrors drv/ioctl.h) ----------------- */

#define NVM_CAPS_VERSION_1            1u

#define NVM_CAP_DMABUF_V1             (1ull << 0)
#define NVM_CAP_DMABUF_V2             (1ull << 1)
#define NVM_CAP_DMABUF_REQUIRE_P2P    (1ull << 2)
#define NVM_CAP_DMABUF_MAPPING_CLASS  (1ull << 3)
#define NVM_CAP_DMABUF_PIN_ACCOUNTING (1ull << 4)

struct nvm_ioctl_caps {
    __u32  struct_size;
    __u16  version;
    __u16  kernel_uapi_version;
    __u64  flags;
    __u64  max_dmabuf_map_bytes;
    __u64  per_file_pin_limit_bytes;
    __u64  global_pin_limit_bytes;
    __u64  current_file_pinned_bytes;
    __u64  current_global_pinned_bytes;
    __u64  reserved[4];
};


/* Supported operations */
enum nvm_ioctl_type
{
    NVM_MAP_HOST_MEMORY         = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY       = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
    NVM_UNMAP_MEMORY            = _IOW(NVM_IOCTL_TYPE, 3, uint64_t),
    NVM_MAP_DMABUF_MEMORY       = _IOWR(NVM_IOCTL_TYPE, 4, struct nvm_ioctl_dmabuf),
    NVM_REGISTER_INTERRUPT      = _IOW(NVM_IOCTL_TYPE, 5, struct nvm_ioctl_irq),
    NVM_UNREGISTER_INTERRUPT    = _IOW(NVM_IOCTL_TYPE, 6, struct nvm_ioctl_irq),
    NVM_GET_NUM_VECTORS         = _IOR(NVM_IOCTL_TYPE, 7, __u32),
    NVM_GET_CAPS                = _IOWR(NVM_IOCTL_TYPE, 8, struct nvm_ioctl_caps),
    NVM_MAP_DMABUF_MEMORY_V2    = _IOWR(NVM_IOCTL_TYPE, 9, struct nvm_ioctl_dmabuf_v2),
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */
