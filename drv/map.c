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

#ifdef _CUDA
#include <nv-p2p.h>

struct gpu_region
{
    nvidia_p2p_page_table_t* pages;
    nvidia_p2p_dma_mapping_t** mappings;
};
#endif


#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)

uint32_t max_num_ctrls = 64;


static struct map* create_descriptor(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map* map = NULL;

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
    atomic_set(&map->invalid, 0);


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
    struct map* result = NULL;

    /* Acquire refcounted pid once. get_task_pid returns a referenced
     * pointer (via get_pid), so we must put_pid before returning. */
    struct pid* current_pid = get_task_pid(current, PIDTYPE_TGID);

    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        /* Compare refcounted pid pointers, not numeric PID values,
         * to prevent collisions after PID/TGID reuse on long-running
         * systems. Pointer comparison is safe because the struct pid
         * is only freed after all refs are dropped (RCU + refcount). */
        if (map->owner_pid == current_pid)
        {
            /* Match address using the mapping's own page size.
             * Previously the unconditional GPU_PAGE_MASK (64 KiB)
             * could cause HIP-vs-HIP false matches when two 4 KiB
             * mappings resided in the same 64 KiB region. Each
             * mapping is now matched only by its own page_size. */
            if (map->page_size > 0 &&
                map->vaddr == (vaddr & ~((u64)map->page_size - 1)))
            {
                result = map;
                break;
            }
        }

        element = list_next(element);
    }

    if (current_pid != NULL)
        put_pid(current_pid);

    return result;
}



/* Atomically find and remove a mapping from the list under the
 * list spinlock. The caller is responsible for calling
 * unmap_and_release() on the returned map. */
struct map* map_find_and_remove(struct list* list, u64 vaddr)
{
    struct map* result = NULL;
    struct pid* current_pid;

    current_pid = get_task_pid(current, PIDTYPE_TGID);

    spin_lock(&list->lock);
    {
        struct list_node* element = list_next(&list->head);
        while (element != NULL)
        {
            struct map* map = container_of(element, struct map, list);

            if (map->owner_pid == current_pid)
            {
                if (map->page_size > 0 &&
                    map->vaddr == (vaddr & ~((u64)map->page_size - 1)))
                {
                    /* Detach from list inline (already holding lock).
                     * list_remove would try to re-lock and deadlock. */
                    element->prev->next = element->next;
                    element->next->prev = element->prev;
                    element->list = NULL;
                    element->next = NULL;
                    element->prev = NULL;
                    result = map;
                    break;
                }
            }
            element = list_next(element);
        }
    }
    spin_unlock(&list->lock);

    if (current_pid != NULL)
        put_pid(current_pid);

    return result;
}



static void release_user_pages(struct map* map)
{
    unsigned long i;
    struct page** pages;
    struct device* dev;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    pages = (struct page**) map->data;
    for (i = 0; i < map->n_addrs; ++i)
    {
        put_page(pages[i]);
    }

    kvfree(map->data);
    map->data = NULL;

    //printk(KERN_DEBUG "Released %lu host pages\n", map->n_addrs);
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
#elif LINUX_VERSION_CODE < KERNEL_VERSION(6, 5, 0)
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL);
#else
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages);
#endif
    if (retval <= 0)
    {
        kfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval;
    }

    if (map->n_addrs != retval)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %ld\n", map->n_addrs, retval);
    }
    map->n_addrs = retval;
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
            return retval;
        }
       // printk("map_user_page: device: %02x:%02x.%1x\tvaddr: %llx\ti: %lu\tdma_addr: %llx\n", map->pdev->bus->number, PCI_SLOT(map->pdev->devfn), PCI_FUNC(map->pdev->devfn), (uint64_t) map->vaddr, i, map->addrs[i]);
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
    struct gpu_region* gd = (struct gpu_region*) map->data;
    struct list* list = map->ctrl_list;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            const struct list_node* element = list_next(&list->head);
            struct ctrl* ctrl;

            uint32_t j = 0;
            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);
                if (gd->mappings[j] != NULL)
                    nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                element = list_next(element);
            }
            kfree(gd->mappings);

        }

        if (gd->pages != NULL)
        {
            nvidia_p2p_free_page_table(gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    unmap_and_release(map);
}
#endif



#ifdef _CUDA
void release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;
    struct list* list = map->ctrl_list;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            const struct list_node* element = list_next(&list->head);
            struct ctrl* ctrl;

            uint32_t j = 0;
            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);
                if (gd->mappings[j] != NULL)
                    nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                element = list_next(element);
            }
            kfree(gd->mappings);

        }

        if (gd->pages != NULL)
        {
            nvidia_p2p_put_pages(0, 0, map->vaddr, gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        //printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
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

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    gd->mappings = (nvidia_p2p_dma_mapping_t**)  kmalloc(sizeof(nvidia_p2p_dma_mapping_t*) * max_num_ctrls, GFP_KERNEL);
    
    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        kfree(gd);
        return -ENOMEM;
    }
    for (j = 0; j < max_num_ctrls; j++)
        gd->mappings[j] = NULL;

    gd->pages = NULL;
    //gd->mappings = NULL;

    map->page_size = GPU_PAGE_SIZE;
    map->data = gd;
    map->release = release_gpu_memory;

    err = nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
            (void (*)(void*)) force_release_gpu_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "nvidia_p2p_get_pages() failed: %d\n", err);
        return err;
    }

    /* Debug: print raw physical_address from nvidia_p2p_page_table_t */
#if 0
    {
        int dbg_max = (gd->pages->entries < 4) ? gd->pages->entries : 4;
        int d;
        printk(KERN_INFO "BAM P2P: entries=%u page_size=%u\n",
               gd->pages->entries, gd->pages->page_size);
        for (d = 0; d < dbg_max; d++) {
            printk(KERN_INFO "BAM P2P: physical_address[%d] = 0x%llx\n",
                   d, (unsigned long long)gd->pages->pages[d]->physical_address);
        }
        if (gd->pages->entries > 4) {
            printk(KERN_INFO "BAM P2P: physical_address[%u] = 0x%llx (last)\n",
                   gd->pages->entries - 1,
                   (unsigned long long)gd->pages->pages[gd->pages->entries - 1]->physical_address);
        }
    }
#endif

    element = list_next(&list->head);


    j = 0;
    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        err = nvidia_p2p_dma_map_pages(ctrl->pdev, gd->pages, gd->mappings + (j++));
        if (err != 0)
        {
            //printk(KERN_ERR "nvidia_p2p_dma_map_pages() failed for nvme%u: %d\n", j-1, err);
            return err;
        }
        //for (i = 0; i < map->n_addrs; ++i)
        //{

        //   printk("device: %u\ti: %lu\tpaddr: %llx\n", (j-1), i, (uint64_t)  gd->mappings[j-1]->dma_addresses[i]);
        //}
        if (j == 1) {
            for (i = 0; i < map->n_addrs; ++i)
            {
                map->addrs[i] = gd->mappings[0]->dma_addresses[i];
                //printk("++paddr: %llx\n", (uint64_t) map->addrs[i]);
            }
        }
        element = list_next(element);
    }




    if (map->n_addrs != gd->pages->entries)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, gd->pages->entries);
    }

    map->n_addrs = gd->pages->entries;

    /* Debug: side-by-side comparison of p2p physical vs DMA addresses */
#if 0
    {
        int cmp_max = (map->n_addrs < 4) ? map->n_addrs : 4;
        int c;
        printk(KERN_INFO "BAM P2P: === physical_address vs dma_address comparison ===\n");
        for (c = 0; c < cmp_max; c++) {
            printk(KERN_INFO "BAM P2P: [%d] p2p_phys=0x%llx  dma=0x%llx  diff=0x%llx\n",
                   c,
                   (unsigned long long)gd->pages->pages[c]->physical_address,
                   (unsigned long long)map->addrs[c],
                   (unsigned long long)(map->addrs[c] - gd->pages->pages[c]->physical_address));
        }
    }
#endif
    
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

#if defined(UGDS_HAVE_DMABUF)
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

    /* Map for DMA -- dynamic importer must hold reservation lock */
    dma_resv_lock(dr->dmabuf->resv, NULL);
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
