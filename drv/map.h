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


typedef void (*release)(struct map*);


/*
 * Describes a range of mapped memory.
 */
struct map
{
    struct list_node    list;           /* Linked list header */
    struct task_struct* owner;          /* Owner of mapping */
    u64                 vaddr;          /* Starting virtual address */
    struct list*        ctrl_list;
    struct pci_dev*     pdev;           /* Reference to physical PCI device */
    unsigned long       page_size;      /* Logical page size */
    void*               data;           /* Custom data */
    release             release;        /* Custom callback for unmapping and releasing memory */
    unsigned long       n_addrs;        /* Number of mapped pages */
    atomic_t            invalid;        /* Set by dmabuf move_notify */
    uint64_t            addrs[1];       /* Bus addresses */
};



/*
 * Lock and map userspace pages for DMA.
 */
struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages);



/*
 * Unmap and release memory.
 */
void unmap_and_release(struct map* map);



#ifdef _CUDA
/*
 * Lock and map GPU device memory.
 */
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list);
#endif



#if defined(UGDS_HAVE_DMABUF)
/*
 * Map GPU memory via standard Linux DMA-buf framework.
 * Used by AMD HIP/ROCm backend and CUDA dma-buf export.
 */
struct map* map_dmabuf(struct list* list, const struct ctrl* ctrl,
                        u64 gpu_ptr, int dmabuf_fd,
                        u64 dmabuf_offset, unsigned long n_pages,
                        size_t ioaddrs_capacity);
#endif



/*
 * Find memory mapping from vaddr and current task
 */
struct map* map_find(const struct list* list, u64 vaddr);


#if defined(UGDS_HAVE_DMABUF)
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
