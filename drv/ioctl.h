/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#ifndef __UGDS_DRV_IOCTL_H__
#define __UGDS_DRV_IOCTL_H__
#ifdef __linux__

#include <linux/types.h>
#include <asm/ioctl.h>

#define NVM_IOCTL_TYPE          0x80

struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
};

#ifdef _HIP
struct nvm_ioctl_dmabuf
{
    __u64  gpu_ptr;           /* Original GPU VA -- unmap identity */
    __s32  dmabuf_fd;         /* DMA-buf fd from HSA export */
    __u32  __pad;             /* Alignment */
    __u64  dmabuf_offset;     /* Offset within dmabuf allocation */
    __u64  size;              /* Total buffer size in bytes */
    __u64  ioaddrs_capacity;  /* Max entries in ioaddrs buffer */
    __u64  ioaddrs;           /* Output: DMA bus addresses (__user pointer) */
};
#endif

enum nvm_ioctl_type
{
    NVM_MAP_HOST_MEMORY         = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
#ifdef _CUDA
    NVM_MAP_DEVICE_MEMORY       = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
#endif
    NVM_UNMAP_MEMORY            = _IOW(NVM_IOCTL_TYPE, 3, uint64_t),
#ifdef _HIP
    NVM_MAP_DMABUF_MEMORY       = _IOWR(NVM_IOCTL_TYPE, 4, struct nvm_ioctl_dmabuf),
#endif
};

#endif /* __linux__ */
#endif /* __UGDS_DRV_IOCTL_H__ */
