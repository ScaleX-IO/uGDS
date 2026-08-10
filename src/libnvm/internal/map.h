#ifndef __NVM_INTERNAL_LINUX_MAP_H__
#define __NVM_INTERNAL_LINUX_MAP_H__
#ifdef __linux__

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
    MAP_TYPE_DMABUF_CUDA =   0x10
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
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_MAP_H__ */
