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
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */
