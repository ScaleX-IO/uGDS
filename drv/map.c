/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#include "map.h"
#include "list.h"
#include "ctrl.h"
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/pci.h>

#ifdef _CUDA
#include <nv-p2p.h>

struct gpu_region
{
    nvidia_p2p_page_table_t* pages;
    nvidia_p2p_dma_mapping_t** mappings;
    uint32_t n_mappings;       /* count of mappings array */
    struct pci_dev** pdevs;    /* pdev snapshot for safe teardown */
    u64 vaddr;                 /* GPU virtual address for put_pages */
};

/* Callback-safe cleanup: unmap DMA mappings via free_dma_mapping
 * (deferred-safe) and free the page table via free_page_table.
 * Called from the NVIDIA invalidation callback only. */
static void gpu_region_free_callback(struct gpu_region* gd)
{
    uint32_t j;
    if (gd == NULL)
        return;

    if (gd->mappings != NULL)
    {
        for (j = 0; j < gd->n_mappings; j++)
        {
            if (gd->mappings[j] != NULL)
                nvidia_p2p_free_dma_mapping(gd->mappings[j]);
        }
        kfree(gd->mappings);
    }

    if (gd->pdevs != NULL)
    {
        for (j = 0; j < gd->n_mappings; j++)
            pci_dev_put(gd->pdevs[j]);
        kfree(gd->pdevs);
    }

    if (gd->pages != NULL)
        nvidia_p2p_free_page_table(gd->pages);

    kfree(gd);
}

/* Normal cleanup: unmap DMA mappings via dma_unmap_pages and free
 * the page table via put_pages. Called from release_gpu_memory and
 * setup error paths (before the callback can fire, i.e. before
 * map->data is published). */
static void gpu_region_free_normal(struct gpu_region* gd)
{
    uint32_t j;
    if (gd == NULL)
        return;

    if (gd->mappings != NULL)
    {
        for (j = 0; j < gd->n_mappings; j++)
        {
            if (gd->mappings[j] != NULL)
                nvidia_p2p_dma_unmap_pages(gd->pdevs[j], gd->pages, gd->mappings[j]);
        }
        kfree(gd->mappings);
    }

    if (gd->pdevs != NULL)
    {
        for (j = 0; j < gd->n_mappings; j++)
            pci_dev_put(gd->pdevs[j]);
        kfree(gd->pdevs);
    }

    if (gd->pages != NULL)
        nvidia_p2p_put_pages(0, 0, gd->vaddr, gd->pages);

    kfree(gd);
}
#endif


#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)


static struct map* create_descriptor(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map* map = NULL;

    /* Prevent integer overflow in the flexible-array allocation.
     * (n_pages - 1) * sizeof(uint64_t) must not overflow unsigned long. */
    if (n_pages == 0 || n_pages > (ULONG_MAX - sizeof(struct map)) / sizeof(uint64_t) + 1)
    {
        printk(KERN_ERR "n_pages overflow: %lu\n", n_pages);
        return ERR_PTR(-EINVAL);
    }

    map = kvmalloc(sizeof(struct map) + (n_pages - 1) * sizeof(uint64_t), GFP_KERNEL);
    if (map == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&map->list);

    map->owner_pid = get_task_pid(current, PIDTYPE_TGID);
    map->vaddr = vaddr;
    map->pdev = ctrl->pdev;
    map->page_size = 0;
    map->data = NULL;
    map->release = NULL;
    map->n_addrs = n_pages;
    map->n_dma_mapped = 0;
    atomic_set(&map->invalid, 0);
    atomic_set(&map->refcount, 1);


    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = 0;
    }

    return map;
}



void unmap_and_release(struct map* map)
{
    list_remove(&map->list);

    if (map->release != NULL && map->data != NULL)
    {
        map->release(map);
    }

    if (map->owner_pid != NULL)
        put_pid(map->owner_pid);

    kvfree(map);
}



struct map* map_find(const struct list* list, u64 vaddr)
{
    const struct list_node* element = list_next(&list->head);
    struct map* map = NULL;
    struct map* exact = NULL;
    struct map* masked = NULL;

    /* Acquire refcounted pid once. get_task_pid returns a referenced
     * pointer (via get_pid), so we must put_pid before returning. */
    struct pid* current_pid = get_task_pid(current, PIDTYPE_TGID);

    /* Two-pass lookup: prefer exact vaddr match, fall back to
     * page-mask match. This prevents a 64 KiB CUDA mapping from
     * shadowing a 4 KiB dma-buf mapping in the same region. */
    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->owner_pid == current_pid)
        {
            if (map->vaddr == vaddr)
            {
                exact = map;
                break;
            }
            if (masked == NULL &&
                map->page_size > 0 &&
                map->vaddr == (vaddr & ~((u64)map->page_size - 1)))
            {
                masked = map;
            }
        }

        element = list_next(element);
    }

    if (current_pid != NULL)
        put_pid(current_pid);

    return exact ? exact : masked;
}



/*
 * Helper: unlink a list_node from the doubly-linked list.
 * Must be called with the list spinlock held.
 */
static void list_unlink(struct list_node* element)
{
    element->prev->next = element->next;
    element->next->prev = element->prev;
    element->list = NULL;
    element->next = NULL;
    element->prev = NULL;
}

/*
 * Helper: find a mapping by exact vaddr or page-masked vaddr
 * for the current task and matching PCI device. Returns the map or NULL.
 * Must be called with the list spinlock held.
 *
 * The pdev match prevents cross-device reuse: two NVMe controllers
 * in the same process must not share DMA addresses.
 */
static struct map* map_lookup_locked(const struct list* list,
                                      u64 vaddr,
                                      const struct pci_dev* pdev,
                                      unsigned long n_pages,
                                      unsigned long page_size,
                                      struct list_node** out_element)
{
    struct list_node* element = list_next(&list->head);
    struct map* exact = NULL;
    struct map* masked = NULL;
    struct list_node* masked_element = NULL;
    struct pid* current_pid = get_task_pid(current, PIDTYPE_TGID);

    while (element != NULL)
    {
        struct map* map = container_of(element, struct map, list);

        if (map->owner_pid == current_pid &&
            (pdev == NULL || map->pdev == pdev) &&
            (n_pages == 0 || map->n_addrs == n_pages) &&
            (page_size == 0 || map->page_size == page_size))
        {
            if (map->vaddr == vaddr)
            {
                exact = map;
                if (out_element)
                    *out_element = element;
                break;
            }
            if (masked == NULL &&
                map->page_size > 0 &&
                map->vaddr == (vaddr & ~((u64)map->page_size - 1)))
            {
                masked = map;
                masked_element = element;
            }
        }
        element = list_next(element);
    }

    if (current_pid != NULL)
        put_pid(current_pid);

    if (exact)
        return exact;

    if (masked)
    {
        if (out_element)
            *out_element = masked_element;
        return masked;
    }

    if (out_element)
        *out_element = NULL;
    return NULL;
}

/*
 * Atomically find an existing mapping and increment its refcount.
 * Matches by (owner_pid, vaddr, pdev). Returns the existing map,
 * or NULL if not found.
 */
struct map* map_find_and_ref(struct list* list, u64 vaddr,
                             const struct pci_dev* pdev,
                             unsigned long n_pages,
                             unsigned long page_size)
{
    struct map* result = NULL;

    spin_lock(&list->lock);
    {
        result = map_lookup_locked(list, vaddr, pdev, n_pages, page_size, NULL);
        if (result != NULL)
        {
            int old = atomic_read(&result->refcount);
            if (old >= 0x7FFFFFFF)
            {
                spin_unlock(&list->lock);
                return ERR_PTR(-EMFILE);
            }
            atomic_inc(&result->refcount);
        }
    }
    spin_unlock(&list->lock);

    return result;
}

/*
 * Decrement a mapping's refcount. If it reaches 0 the mapping is
 * removed from the list and freed.
 *
 * The caller must hold map_create_mutex (or otherwise guarantee no
 * concurrent force_release) to prevent a race with the NVIDIA P2P
 * callback path.
 */
void map_put_ref(struct map* map)
{
    struct list* lst;

    if (map == NULL)
        return;

    lst = map->list.list;
    spin_lock(&lst->lock);
    {
        if (atomic_dec_and_test(&map->refcount))
        {
            list_unlink(&map->list);
        }
        else
        {
            map = NULL; /* still referenced, don't free */
        }
    }
    spin_unlock(&lst->lock);

    if (map != NULL)
        unmap_and_release(map);
}

/*
 * Atomically find and remove (or decrement) a mapping from the list.
 *
 * n_pages=0 and page_size=0 act as wildcards. This is correct for
 * UNMAP because userspace should never create two maps with the same
 * VA on the same controller with different backends; if it does, the
 * exact-match path returns the first matching map deterministically.
 *
 * Returns:
 *   - pointer to the removed map if refcount drops to 0 (caller must
 *     unmap_and_release)
 *   - ERR_PTR(-EAGAIN) if the mapping was found but still referenced
 *     (refcount decremented, not removed)
 *   - NULL if not found
 */
struct map* map_find_and_remove(struct list* list, u64 vaddr,
                                const struct pci_dev* pdev)
{
    struct map* result = NULL;
    struct list_node* target_element = NULL;

    spin_lock(&list->lock);
    {
        result = map_lookup_locked(list, vaddr, pdev, 0, 0, &target_element);
        if (result != NULL && target_element != NULL)
        {
            if (atomic_dec_and_test(&result->refcount))
            {
                /* refcount reached 0: safe to remove */
                list_unlink(target_element);
            }
            else
            {
                /* Still in use by another on-the-fly user */
                result = ERR_PTR(-EAGAIN);
            }
        }
        else
        {
            result = NULL;
        }
    }
    spin_unlock(&list->lock);

    return result;
}



/*
 * Check if any mapping owned by the current task overlaps the
 * byte range [vaddr, vaddr + n_bytes) on the given list and pdev.
 * Uses proper range-overlap detection so mappings at different
 * page granularities (e.g. 4 KiB dmabuf vs 64 KiB CUDA) are
 * correctly detected even when their base addresses differ.
 *
 * Must be called with map_create_mutex held (serializes against
 * concurrent MAP/UNMAP that could add/remove entries).
 */
bool map_range_exists(struct list* list, u64 vaddr, u64 n_bytes,
                      const struct pci_dev* pdev)
{
    struct list_node* element;
    struct pid* current_pid;
    bool found = false;
    u64 new_end;

    /* Reject wrapping ranges from user-controlled addresses.
     * Return true so the caller rejects with -EEXIST rather than
     * allowing a nonsensical wrapped range to be created. */
    if (vaddr > U64_MAX - n_bytes)
        return true;

    new_end = vaddr + n_bytes;

    current_pid = get_task_pid(current, PIDTYPE_TGID);

    spin_lock(&list->lock);
    {
        element = list_next(&list->head);
        while (element != NULL)
        {
            struct map* map = container_of(element, struct map, list);
            u64 existing_end;

            if (map->owner_pid != current_pid ||
                map->pdev != pdev)
            {
                element = list_next(element);
                continue;
            }

            existing_end = map->vaddr + (u64)map->n_addrs * (u64)map->page_size;

            /* Standard interval overlap: [a, b) vs [c, d) */
            if (vaddr < existing_end && new_end > map->vaddr)
            {
                found = true;
                break;
            }

            element = list_next(element);
        }
    }
    spin_unlock(&list->lock);

    if (current_pid != NULL)
        put_pid(current_pid);

    return found;
}


static void release_user_pages(struct map* map)
{
    unsigned long i;
    struct page** pages;
    struct device* dev;

    dev = &map->pdev->dev;
    /* Only unmap entries that were successfully DMA-mapped. */
    for (i = 0; i < map->n_dma_mapped; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    pages = (struct page**) map->data;
    if (pages != NULL)
    {
        for (i = 0; i < map->n_addrs; ++i)
        {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
            unpin_user_page(pages[i]);
#else
            put_page(pages[i]);
#endif
        }
    }

    kvfree(map->data);
    map->data = NULL;
}



static long map_user_pages(struct map* map)
{
    unsigned long i;
    long retval;
    struct page** pages;
    struct device* dev;

    pages = (struct page**) kvcalloc(map->n_addrs, sizeof(struct page*), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_CRIT "Failed to allocate page array\n");
        return -ENOMEM;
    }

#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 5, 7)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(current, current->mm, map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4, 8, 17)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#elif LINUX_VERSION_CODE < KERNEL_VERSION(5, 6, 0)
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL);
#elif LINUX_VERSION_CODE < KERNEL_VERSION(6, 5, 0)
    retval = pin_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE | FOLL_LONGTERM, pages, NULL);
#else
    retval = pin_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE | FOLL_LONGTERM, pages);
#endif
    if (retval <= 0)
    {
        kvfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval;
    }

    if (map->n_addrs != retval)
    {
        /* Partial GUP success: the ioctl ABI returns a fixed-size
         * buffer and does not communicate the actual mapped count.
         * Returning partial mappings leaves stale/zero DMA addresses
         * that userspace would pass to NVMe DMA. Fail hard instead. */
        unsigned long g;
        for (g = 0; g < retval; g++)
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
            unpin_user_page(pages[g]);
#else
            put_page(pages[g]);
#endif
        kvfree(pages);
        printk(KERN_ERR "get_user_pages() got %ld of %lu pages\n", retval, map->n_addrs);
        return -ENOMEM;
    }
    map->page_size = PAGE_SIZE;
    map->data = (void*) pages;
    map->release = release_user_pages;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        retval = dma_mapping_error(dev, map->addrs[i]);
        if (retval != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            /* n_dma_mapped tracks how many were successfully mapped,
             * so release_user_pages only unmaps those entries. */
            return retval;
        }
        map->n_dma_mapped++;
    }

    return 0;
}



struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    long err;
    struct map* md;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = PAGE_SIZE;

    err = map_user_pages(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu host pages starting at address %llx\n", 
    //        md->n_addrs, md->vaddr);
    return md;
}



#ifdef _CUDA
static void force_release_gpu_memory(struct map* map)
{
    /* NVIDIA force-reclaim callback. Runs in process context.
     *
     * Atomically claims gd via xchg. If we get gd, we fully own it
     * and free it using the callback-safe path (free_dma_mapping +
     * free_page_table). If we get NULL, setup hasn't published yet
     * or the normal release path already took it.
     *
     * Once published, the setup thread checks map->invalid at every
     * iteration and on error. If it sees invalid, it knows the
     * callback has already claimed gd and must not touch it. */

    struct gpu_region* gd;
    unsigned long n_addrs;

    n_addrs = map->n_addrs;
    atomic_set(&map->invalid, 1);

    gd = (struct gpu_region*) xchg(&map->data, NULL);

    if (gd != NULL)
    {
        gpu_region_free_callback(gd);
        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", n_addrs);
    }
}
#endif



#ifdef _CUDA
void release_gpu_memory(struct map* map)
{
    /* Normal unregister path. Atomically claim gd. If we win,
     * free via the normal path (dma_unmap_pages + put_pages).
     * If we lose, the callback already took and freed it.
     *
     * put_pages blocks until any outstanding callback completes. */

    struct gpu_region* gd = (struct gpu_region*) xchg(&map->data, NULL);

    if (gd != NULL)
        gpu_region_free_normal(gd);
}
#endif



#ifdef _CUDA
int map_gpu_memory(struct map* map, struct list* list)
{
    unsigned long i;
    uint32_t j;
    int err;
    struct gpu_region* gd;
    const struct list_node* element;
    struct ctrl* ctrl;
    uint32_t n_ctrls = 0;

    /* Snapshot controllers under spinlock: count + collect pdevs
     * in a single pass to prevent TOCTOU between count and fill. */
    spin_lock(&list->lock);
    {
        element = list_next(&list->head);
        while (element != NULL)
        {
            n_ctrls++;
            element = list_next(element);
        }
    }
    spin_unlock(&list->lock);

    if (n_ctrls == 0)
        return -ENODEV;

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    gd->mappings = (nvidia_p2p_dma_mapping_t**) kmalloc(sizeof(nvidia_p2p_dma_mapping_t*) * n_ctrls, GFP_KERNEL);
    
    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        kfree(gd);
        return -ENOMEM;
    }

    gd->pdevs = (struct pci_dev**) kmalloc(sizeof(struct pci_dev*) * n_ctrls, GFP_KERNEL);
    if (gd->pdevs == NULL)
    {
        printk(KERN_CRIT "Failed to allocate pdev snapshot\n");
        kfree(gd->mappings);
        kfree(gd);
        return -ENOMEM;
    }

    gd->n_mappings = n_ctrls;

    for (j = 0; j < n_ctrls; j++)
        gd->mappings[j] = NULL;

    gd->pages = NULL;
    gd->vaddr = map->vaddr;

    /* Collect pdev snapshot under spinlock and pin each device. */
    spin_lock(&list->lock);
    {
        j = 0;
        element = list_next(&list->head);
        while (element != NULL && j < n_ctrls)
        {
            ctrl = container_of(element, struct ctrl, list);
            gd->pdevs[j] = pci_dev_get(ctrl->pdev);
            j++;
            element = list_next(element);
        }
    }
    spin_unlock(&list->lock);

    /* If list shrank between count and collect, abort. */
    if (j < n_ctrls)
    {
        uint32_t k;
        for (k = 0; k < j; k++)
            pci_dev_put(gd->pdevs[k]);
        kfree(gd->pdevs);
        kfree(gd->mappings);
        kfree(gd);
        return -ENODEV;
    }

    map->page_size = GPU_PAGE_SIZE;
    map->release = release_gpu_memory;
    /* map->data stays NULL until setup completes (deferred publish).
     *
     * If the NVIDIA invalidation callback fires during setup, it
     * gets NULL from xchg and returns without touching gd. The
     * setup thread detects invalid and calls gpu_region_free_normal,
     * which uses dma_unmap_pages + put_pages.
     *
     * This is correct per the NVIDIA P2P API contract: when the
     * callback gets NULL (setup still in progress), the page table
     * has NOT been revoked. The setup thread still owns gd and can
     * safely call put_pages to unregister. put_pages blocks until
     * the callback has completed, ensuring no UAF.
     *
     * The callback-safe APIs (free_dma_mapping, free_page_table) are
     * ONLY used from within the callback itself (gpu_region_free_callback),
     * for the case where the callback wins xchg and takes ownership. */

    err = nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
            (void (*)(void*)) force_release_gpu_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "nvidia_p2p_get_pages() failed: %d\n", err);
        gpu_region_free_normal(gd);
        return err;
    }

    /* Validate entries. */
    if (gd->pages->entries != map->n_addrs)
    {
        printk(KERN_ERR "nvidia_p2p_get_pages returned %u entries, expected %lu\n",
               gd->pages->entries, map->n_addrs);
        gpu_region_free_normal(gd);
        return -EIO;
    }

    /* If callback already fired, put_pages (inside free_normal) syncs. */
    if (atomic_read(&map->invalid))
    {
        gpu_region_free_normal(gd);
        return -EIO;
    }

    /* DMA-map pages using the snapshotted pdevs. */
    {
        bool found_request_pdev = false;
        j = 0;
        while (j < n_ctrls)
        {
            /* Check for force_release between iterations. If the
             * callback fired, the page table is revoked. Abort and
             * clean up via put_pages (syncs with callback). */
            if (atomic_read(&map->invalid))
            {
                gpu_region_free_normal(gd);
                return -EIO;
            }

            err = nvidia_p2p_dma_map_pages(gd->pdevs[j], gd->pages, gd->mappings + j);
            if (err != 0)
            {
                /* gpu_region_free_normal unmaps all non-NULL mappings
                 * and calls put_pages (syncs with callback). */
                gpu_region_free_normal(gd);
                return err;
            }
            /* Copy DMA addresses from the mapping for the caller's pdev. */
            if (gd->pdevs[j] == map->pdev)
            {
                found_request_pdev = true;
                for (i = 0; i < map->n_addrs; ++i)
                    map->addrs[i] = gd->mappings[j]->dma_addresses[i];
            }
            j++;
        }

        if (!found_request_pdev)
        {
            gpu_region_free_normal(gd);
            return -ENODEV;
        }
    }

    /* Publish gd AFTER all setup completes. */
    map->data = gd;

    /* If callback fired during setup, it took gd via xchg and freed
     * it via gpu_region_free_callback. The callback saves n_addrs
     * as a local before xchg, so it never dereferences map after
     * xchg. It is safe for the caller to unmap_and_release (kvfree)
     * map even if the callback hasn't fully returned. */
    if (atomic_read(&map->invalid))
        return -EIO;

    return 0;
}
#endif



#ifdef _CUDA
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list)
{
    int err;
    struct map* md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = GPU_PAGE_SIZE;
    md->ctrl_list = ctrl_list;
    err = map_gpu_memory(md, ctrl_list);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n", 
    //        md->n_addrs, md->vaddr);
    return md;
}
#endif



/* - AMD HIP / DMA-buf backend ------------------ */

#ifdef _HIP
#include <linux/dma-buf.h>
#include <linux/scatterlist.h>
#include <linux/dma-resv.h>

struct dmabuf_region
{
    struct dma_buf*            dmabuf;
    struct dma_buf_attachment* attachment;
    struct sg_table*           sgt;
};


/*
 * move_notify callback -- called by the exporter if it needs to move the buffer.
 * With dma_buf_pin() held for the mapping lifetime, the exporter cannot move
 * the buffer, so this callback should never fire. If it does, we set the
 * invalid flag. The setup-time check in pci.c returns -EIO if invalid is
 * set before addresses are copied to userspace.
 */
static void ugds_dmabuf_move_notify(struct dma_buf_attachment* attachment)
{
    struct map* map = (struct map*) attachment->importer_priv;

    if (map)
    {
        atomic_set(&map->invalid, 1);
        WARN_ONCE(1, "uGDS: dmabuf move_notify -- mapping %llx invalidated. "
                     "Pinned VRAM must not migrate.\n", map->vaddr);
    }
}


static const struct dma_buf_attach_ops ugds_dmabuf_attach_ops = {
    .allow_peer2peer = true,
    .move_notify     = ugds_dmabuf_move_notify,
};


void release_dmabuf_memory(struct map* map)
{
    struct dmabuf_region* dr = (struct dmabuf_region*) map->data;

    if (dr != NULL)
    {
        if (dr->sgt != NULL) {
            dma_resv_lock(dr->dmabuf->resv, NULL);
            dma_buf_unmap_attachment(dr->attachment, dr->sgt, DMA_BIDIRECTIONAL);
            dma_buf_unpin(dr->attachment);
            dma_resv_unlock(dr->dmabuf->resv);
        }
        if (dr->attachment != NULL)
            dma_buf_detach(dr->dmabuf, dr->attachment);
        if (dr->dmabuf != NULL)
            dma_buf_put(dr->dmabuf);
        kfree(dr);
        map->data = NULL;
    }
}


/*
 * Flatten an SG table into per-page DMA addresses.
 * Consumes hsa_offset bytes before extracting addresses.
 *
 * Returns 0 on success, -EINVAL if a non-page-granular residual
 * is encountered before expected_pages are filled.
 *
 * If all expected_pages are filled, trailing bytes in the last
 * SG entry are silently ignored.
 */
int sg_flatten_to_addrs(struct sg_table* sgt, u64* addrs,
                        unsigned long expected_pages,
                        unsigned long ctrl_page_size,
                        u64 hsa_offset)
{
    struct scatterlist* sg;
    unsigned long page_idx = 0;
    u64 remaining_offset = hsa_offset;
    int i;

    for_each_sgtable_dma_sg(sgt, sg, i)
    {
        u64 addr = sg_dma_address(sg);
        unsigned int len = sg_dma_len(sg);

        /* Skip SG entries until we consume the full offset */
        if (remaining_offset > 0)
        {
            if (remaining_offset >= len)
            {
                remaining_offset -= len;
                continue;
            }
            addr += remaining_offset;
            len -= remaining_offset;
            remaining_offset = 0;
        }

        while (len >= ctrl_page_size && page_idx < expected_pages)
        {
            /* Reject unaligned DMA addresses. amdgpu P2P should
             * always return page-aligned addresses, but a defensive
             * check prevents a corrupted PRP list. */
            if (addr & (ctrl_page_size - 1))
            {
                printk(KERN_ERR "uGDS: SG entry %u address %llx is not "
                       "%lu-byte aligned\n",
                       i, addr, ctrl_page_size);
                return -EINVAL;
            }
            addrs[page_idx++] = addr;
            addr += ctrl_page_size;
            len -= ctrl_page_size;
        }

        /* Fail on non-page-granular SG residual.
         * A partial page means the DMA address list would be
         * shifted or incomplete, leading to data corruption. */
        if (len > 0 && page_idx < expected_pages)
        {
            printk(KERN_ERR "uGDS: SG entry %u has %u non-page-granular "
                   "residual bytes (page_size=%lu)\n",
                   i, len, ctrl_page_size);
            return -EINVAL;
        }

        if (page_idx >= expected_pages)
            break;
    }

    if (page_idx != expected_pages)
    {
        printk(KERN_ERR "uGDS: dmabuf page count mismatch: got %lu, expected %lu\n",
               page_idx, expected_pages);
        return -EINVAL;
    }

    return 0;
}
#ifdef UGDS_KUNIT
EXPORT_SYMBOL_GPL(sg_flatten_to_addrs);
#endif


static int map_dmabuf_memory(struct map* map, int dmabuf_fd,
                              u64 hsa_offset, unsigned long expected_pages,
                              size_t ioaddrs_capacity)
{
    struct dmabuf_region* dr;
    unsigned long ctrl_page_size = PAGE_SIZE;
    int err;
    long fence_ret;

    if (expected_pages > ioaddrs_capacity)
    {
        printk(KERN_ERR "uGDS: page count %lu exceeds ioaddrs capacity %zu\n",
               expected_pages, ioaddrs_capacity);
        return -EOVERFLOW;
    }

    dr = kmalloc(sizeof(struct dmabuf_region), GFP_KERNEL);
    if (dr == NULL)
    {
        printk(KERN_CRIT "uGDS: failed to allocate dmabuf region\n");
        return -ENOMEM;
    }

    dr->dmabuf = dma_buf_get(dmabuf_fd);
    if (IS_ERR(dr->dmabuf))
    {
        kfree(dr);
        return PTR_ERR(dr->dmabuf);
    }

    /* Cross-check requested range against actual dma-buf size.
     * Use checked arithmetic to prevent overflow. */
    if (expected_pages > dr->dmabuf->size / ctrl_page_size ||
        hsa_offset > dr->dmabuf->size - expected_pages * ctrl_page_size)
    {
        printk(KERN_ERR "uGDS: requested range (%llu + %lu pages) "
               "exceeds dma-buf size %zu\n",
               (unsigned long long)hsa_offset, expected_pages,
               dr->dmabuf->size);
        dma_buf_put(dr->dmabuf);
        kfree(dr);
        return -EINVAL;
    }

    dr->attachment = dma_buf_dynamic_attach(dr->dmabuf, &map->pdev->dev,
                                             &ugds_dmabuf_attach_ops, map);
    if (IS_ERR(dr->attachment))
    {
        dma_buf_put(dr->dmabuf);
        kfree(dr);
        return PTR_ERR(dr->attachment);
    }

    /*
     * Pin the attachment to prevent VRAM migration for the mapping lifetime.
     * dma_buf_pin() must be called with reservation lock held.
     * While pinned, the exporter cannot move the buffer and move_notify
     * will not fire -- DMA addresses stay valid until unmap.
     *
     * Synchronization contract:
     * The pin prevents VRAM migration, but does NOT prevent concurrent
     * GPU writes to the same allocation. The caller (userspace uGDS IO)
     * MUST ensure exclusive access during NVMe DMA transfer -- no concurrent
     * HIP kernel writes to the mapped region while IO is in flight.
     *
     * No dma-resv fence is published because the kernel bypass path does
     * not go through the DMA-buf sync API -- NVMe PRP addresses are used
     * directly from userspace via the NVMe submission queue.
     */
    dma_resv_lock(dr->dmabuf->resv, NULL);
    err = dma_buf_pin(dr->attachment);
    if (!err)
    {
        /* amdgpu may relocate the BO during dma_buf_pin(), firing
         * move_notify once. That settling is expected -- the sgt
         * captured below reflects final pinned addresses. Clear
         * the spurious flag so only genuine post-pin migrations
         * are caught by the fail-stop guard in pci.c. */
        atomic_set(&map->invalid, 0);
    }
    dma_resv_unlock(dr->dmabuf->resv);
    if (err)
    {
        printk(KERN_ERR "uGDS: dma_buf_pin failed: %d\n", err);
        dma_buf_detach(dr->dmabuf, dr->attachment);
        dma_buf_put(dr->dmabuf);
        kfree(dr);
        return err;
    }

    /* Map for DMA -- dynamic importer must hold reservation lock.
     * Wait for any outstanding write fences before capturing DMA
     * addresses, so that GPU writes preceding the import are visible
     * at the DMA-buf backing storage when NVMe accesses it. */
    dma_resv_lock(dr->dmabuf->resv, NULL);
    fence_ret = dma_resv_wait_timeout(dr->dmabuf->resv, DMA_RESV_USAGE_WRITE,
                                 false, MAX_SCHEDULE_TIMEOUT);
    if (fence_ret < 0 && fence_ret != -ERESTARTSYS)
    {
        dma_resv_unlock(dr->dmabuf->resv);
        printk(KERN_ERR "uGDS: dma_resv_wait_timeout failed: %ld\n", fence_ret);
        dma_resv_lock(dr->dmabuf->resv, NULL);
        dma_buf_unpin(dr->attachment);
        dma_resv_unlock(dr->dmabuf->resv);
        dma_buf_detach(dr->dmabuf, dr->attachment);
        dma_buf_put(dr->dmabuf);
        kfree(dr);
        return fence_ret;
    }
    dr->sgt = dma_buf_map_attachment(dr->attachment, DMA_BIDIRECTIONAL);
    dma_resv_unlock(dr->dmabuf->resv);
    if (IS_ERR(dr->sgt))
    {
        err = PTR_ERR(dr->sgt);
        dma_resv_lock(dr->dmabuf->resv, NULL);
        dma_buf_unpin(dr->attachment);
        dma_resv_unlock(dr->dmabuf->resv);
        dma_buf_detach(dr->dmabuf, dr->attachment);
        dma_buf_put(dr->dmabuf);
        kfree(dr);
        return err;
    }

    map->page_size = ctrl_page_size;
    map->data = dr;
    map->release = release_dmabuf_memory;

    /* Flatten sg_table into per-page DMA addresses via helper. */
    err = sg_flatten_to_addrs(dr->sgt, map->addrs, expected_pages,
                              ctrl_page_size, hsa_offset);
    if (err)
        goto fail;

    return 0;

fail:
    dma_resv_lock(dr->dmabuf->resv, NULL);
    dma_buf_unmap_attachment(dr->attachment, dr->sgt, DMA_BIDIRECTIONAL);
    dma_buf_unpin(dr->attachment);
    dma_resv_unlock(dr->dmabuf->resv);
    dma_buf_detach(dr->dmabuf, dr->attachment);
    dma_buf_put(dr->dmabuf);
    kfree(dr);
    map->data = NULL;
    return err;
}


struct map* map_dmabuf(struct list* list, const struct ctrl* ctrl,
                        u64 gpu_ptr, int dmabuf_fd,
                        u64 dmabuf_offset, unsigned long n_pages,
                        size_t ioaddrs_capacity)
{
    int err;
    struct map* md;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    /* Use GPU pointer as vaddr for map_find() lookup on unmap */
    md = create_descriptor(ctrl, gpu_ptr, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = PAGE_SIZE;
    err = map_dmabuf_memory(md, dmabuf_fd, dmabuf_offset, n_pages,
                             ioaddrs_capacity);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    printk(KERN_DEBUG "uGDS: mapped %lu dmabuf pages starting at gpu_ptr %llx\n",
           md->n_addrs, md->vaddr);
    return md;
}
#endif
