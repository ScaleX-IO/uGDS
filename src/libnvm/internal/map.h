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
    MAP_TYPE_CUDA        =   0x1,   // CUDA device memory (NVIDIA)
    MAP_TYPE_HOST        =   0x2,   // Host memory (RAM)
    MAP_TYPE_API         =   0x4,   // Allocated by the API (RAM)
    MAP_TYPE_DMABUF      =   0x8,   // DMA-buf (AMD HIP / standard Linux)
    MAP_TYPE_DMABUF_CUDA =   0x10   // CUDA dma-buf export (NVIDIA via cuMemGetHandleForAddressRange)
};

/* Typed fd close callback for backend-specific dma-buf fd release.
 * Each backend registers its own adapter. Returns 0 on success. */
typedef int (*dmabuf_close_fn)(int fd);



/*
 * Mapping container
 */
struct ioctl_mapping
{
    enum mapping_type   type;   // What kind of memory
    void*               buffer; // GPU pointer (unmap identity) or host buffer
    struct va_range     range;  // Memory range descriptor

    /* DMABUF-specific fields */
    int       dmabuf_fd;        // HSA/CUDA-exported dma-buf file descriptor
    uint64_t  dmabuf_offset;    // Offset within dmabuf allocation

    /* fd ownership + release (RDMA support) */
    bool                retain_fd;   // true: fd retained for RDMA export
    dmabuf_close_fn     close_fn;    // backend-specific close (NULL if N/A)
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_MAP_H__ */
