/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#ifndef __UGDS_DRV_MAP_H__
#define __UGDS_DRV_MAP_H__

#include "list.h"
#include <linux/types.h>
#include <linux/mm_types.h>
#include <linux/atomic.h>


/* Forward declaration */
struct ctrl;
struct map;
struct pci_dev;


typedef void (*release)(struct map*);


/*
 * Describes a range of mapped memory.
 *
 * Ownership: a map is owned by exactly one map_handle in one file's
 * ledger (see pci.c). There is no global map list; all lookup and
 * refcounting happens in the per-file ledger under the file ctx lock.
 */
struct map
{
    u64                 vaddr;          /* Starting virtual address (aligned) */
    struct list*        ctrl_list;
    struct pci_dev*     pdev;           /* Reference to physical PCI device */
    unsigned long       page_size;      /* Logical page size */
    void*               data;           /* Backend data, handed off via xchg */
    release             release;        /* Backend release callback */
    unsigned long       n_addrs;        /* Number of mapped pages */
    unsigned long       n_dma_mapped;   /* Successfully DMA-mapped pages (host backend) */
    atomic_t            invalid;        /* Set by dmabuf move_notify or NVIDIA force_release */
    uint64_t            addrs[1];       /* Bus addresses */
};



/*
 * Lock and map userspace pages for DMA.
 */
struct map* map_userspace(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages);



/*
 * Unmap and release memory.
 */
void unmap_and_release(struct map* map);



#ifdef _CUDA
/*
 * Lock and map GPU device memory.
 */
struct map* map_device_memory(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list);
#endif



#ifdef _HIP
/*
 * Map GPU memory via standard Linux DMA-buf framework.
 * Used by AMD HIP/ROCm backend.
 */
struct map* map_dmabuf(const struct ctrl* ctrl,
                        u64 gpu_ptr, int dmabuf_fd,
                        u64 dmabuf_offset, unsigned long n_pages,
                        size_t ioaddrs_capacity);
#endif



#ifdef _HIP
/* Forward declaration -- avoids pulling <linux/scatterlist.h> into map.h */
struct sg_table;

/*
 * Flatten an SG table into per-page DMA addresses.
 * Called by map_dmabuf_memory() and KUnit tests.
 */
int sg_flatten_to_addrs(struct sg_table* sgt, u64* addrs,
                        unsigned long expected_pages,
                        unsigned long ctrl_page_size,
                        u64 hsa_offset);
#endif


#endif /* __UGDS_DRV_MAP_H__ */
