#ifndef __NVM_INTERNAL_LINUX_MAP_H__
#define __NVM_INTERNAL_LINUX_MAP_H__
#ifdef __linux__

#include <stdint.h>
#include "internal/ioctl.h"
#include "internal/dma.h"


/*
 * What kind of memory are we mapping.
 */
enum mapping_type
{
    MAP_TYPE_CUDA        =   0x1,
    MAP_TYPE_HOST        =   0x2,
    MAP_TYPE_API         =   0x4,
    MAP_TYPE_DMABUF      =   0x8,
    MAP_TYPE_DMABUF_CUDA =   0x10,
    MAP_TYPE_DMABUF_EXT  =   0x20
};

typedef int (*dmabuf_close_fn)(int fd);



/*
 * Mapping container
 */
struct ioctl_mapping
{
    enum mapping_type   type;
    void*               buffer;
    struct va_range     range;

    int       dmabuf_fd;
    uint64_t  dmabuf_offset;

    bool                retain_fd;
    dmabuf_close_fn     close_fn;

    /* V2 dma-buf ioctl parameters (MAP_TYPE_DMABUF_EXT only).
     * When v2_flags is non-zero, linux_device.cpp issues
     * NVM_MAP_DMABUF_MEMORY_V2 instead of the V1 ioctl, and
     * v2_result receives the kernel classification output. */
    uint16_t                    v2_flags;
    struct nvm_ioctl_dmabuf_v2* v2_result;
};


/*
 * Issue the V2 dma-buf map ioctl directly on the controller fd.
 * Defined in linux_device.cpp; called by linux_dma.cpp for the
 * external dma-buf import path that needs classification output.
 * Returns 0 on success or a positive errno.
 */
int _nvm_dmabuf_ioctl_v2(const nvm_ctrl_t* ctrl,
                          struct nvm_ioctl_dmabuf_v2* req);

/*
 * Issue the NVM_GET_CAPS ioctl directly on the controller fd.
 * Returns 0 on success or a positive errno.
 */
int _nvm_dmabuf_get_caps(const nvm_ctrl_t* ctrl,
                          struct nvm_ioctl_caps* caps);


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_MAP_H__ */
