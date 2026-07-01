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

#ifdef _HIP
struct nvm_ioctl_dmabuf
{
    uint64_t  gpu_ptr;           /* Original GPU VA -- unmap identity */
    int32_t   dmabuf_fd;         /* DMA-buf fd from HSA export */
    uint32_t  __pad;             /* Alignment */
    uint64_t  dmabuf_offset;     /* Offset within dmabuf allocation */
    uint64_t  size;              /* Total buffer size in bytes */
    uint64_t  ioaddrs_capacity;  /* Max entries in ioaddrs buffer */
    uint64_t  ioaddrs;           /* Output: DMA bus addresses (pointer) */
};
#endif



/* Supported operations */
enum nvm_ioctl_type
{
    NVM_MAP_HOST_MEMORY         = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY       = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
    NVM_UNMAP_MEMORY            = _IOW(NVM_IOCTL_TYPE, 3, uint64_t),
#ifdef _HIP
    NVM_MAP_DMABUF_MEMORY       = _IOWR(NVM_IOCTL_TYPE, 4, struct nvm_ioctl_dmabuf),
#endif
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */
