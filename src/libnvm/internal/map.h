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
    MAP_TYPE_CUDA   =   0x1,   // CUDA device memory (NVIDIA)
    MAP_TYPE_HOST   =   0x2,   // Host memory (RAM)
    MAP_TYPE_API    =   0x4,   // Allocated by the API (RAM)
    MAP_TYPE_DMABUF =   0x8    // DMA-buf (AMD HIP / standard Linux)
};



/*
 * Mapping container
 */
struct ioctl_mapping
{
    enum mapping_type   type;   // What kind of memory
    void*               buffer; // GPU pointer (unmap identity) or host buffer
    struct va_range     range;  // Memory range descriptor

    /* DMABUF-specific fields */
    int       dmabuf_fd;        // HSA-exported dma-buf file descriptor
    uint64_t  dmabuf_offset;    // Offset within dmabuf allocation
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_MAP_H__ */
