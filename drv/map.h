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
#include <linux/pid.h>
#include <linux/mutex.h>

/* Serialize map creation, removal, and force_release to prevent
 * concurrent MAP/UNMAP/force_release from racing on the same map.
 * Declared in pci.c, used by force_release_gpu_memory() in map.c.
 */
extern struct mutex map_create_mutex;


/* Forward declaration */
struct ctrl;
struct map;
struct pci_dev;


typedef void (*release)(struct map*);


/*
 * Describes a range of mapped memory.
 */
struct map
{
    struct list_node    list;           /* Linked list header */
    struct pid*         owner_pid;      /* Refcounted owner process (prevents PID reuse collision) */
    u64                 vaddr;          /* Starting virtual address */
    struct list*        ctrl_list;
    struct pci_dev*     pdev;           /* Reference to physical PCI device */
    unsigned long       page_size;      /* Logical page size */
    void*               data;           /* Custom data (protected by data_lock) */
    release             release;        /* Custom callback (protected by data_lock) */
    unsigned long       n_addrs;        /* Number of mapped pages */
    unsigned long       n_dma_mapped;   /* Number of successfully DMA-mapped pages (host backend) */
    atomic_t            invalid;        /* Set by dmabuf move_notify or NVIDIA force_release */
    atomic_t            refcount;       /* On-the-fly mapping refcount (prevents premature unmap) */
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



#ifdef _HIP
/*
 * Map GPU memory via standard Linux DMA-buf framework.
 * Used by AMD HIP/ROCm backend.
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

/*
 * Atomically find an existing mapping and increment its refcount.
 * Returns the existing map, or NULL if not found.
 * The list spinlock is held to prevent concurrent races.
 * Used by on-the-fly map creation to deduplicate concurrent
 * maps of the same (pid, vaddr).
 */
struct map* map_find_and_ref(struct list* list, u64 vaddr,
                             const struct pci_dev* pdev,
                             unsigned long n_pages,
                             unsigned long page_size);

/*
 * Decrement a mapping's refcount. If it reaches 0 the mapping is
 * removed from the list and freed.
 *
 * The caller must hold map_create_mutex (or otherwise guarantee no
 * concurrent force_release) to prevent a race with the NVIDIA P2P
 * callback path.
 */
void map_put_ref(struct map* map);

/*
 * Atomically find and remove a mapping from the list.
 * Returns the removed map (caller must unmap_and_release it),
 * NULL if not found, or ERR_PTR(-EAGAIN) if the mapping was
 * found but still referenced by another on-the-fly user
 * (refcount was decremented, not removed). The list spinlock
 * is held during traversal and removal to prevent concurrent
 * races.
 *
 * If pdev is non-NULL, only matches mappings for that PCI device.
 */
struct map* map_find_and_remove(struct list* list, u64 vaddr,
                                const struct pci_dev* pdev);


/*
 * Check if any mapping owned by the current task on the given
 * list overlaps the byte range [vaddr, vaddr + n_bytes).
 * Returns true if an overlap exists (caller should reject
 * the new mapping with -EEXIST).
 */
bool map_range_exists(struct list* list, u64 vaddr, u64 n_bytes,
                      const struct pci_dev* pdev);


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
