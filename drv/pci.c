/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#include "ioctl.h"
#include "list.h"
#include "ctrl.h"
#include "map.h"
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/err.h>
#include <linux/mutex.h>
#include <linux/device.h>
#include <linux/version.h>
#include <linux/uaccess.h>
#include <asm/io.h>
#include <asm/errno.h>
#include <asm/page.h>

#define DRIVER_NAME         "ugds_drv"
#define PCI_CLASS_NVME      0x010802
#define PCI_CLASS_NVME_MASK 0xffffff


MODULE_DESCRIPTION("UserSpace-GDS NVMe DMA helper");
MODULE_LICENSE("Dual BSD/GPL");
#ifdef _HIP
MODULE_IMPORT_NS(DMA_BUF);
#endif
MODULE_VERSION("1.0");


/* Define a filter for selecting devices we are interested in */
static const struct pci_device_id id_table[] = 
{
    { PCI_DEVICE_CLASS(PCI_CLASS_NVME, PCI_CLASS_NVME_MASK) },
    { 0 }
};


/* Reference to the first character device */
static dev_t dev_first;


/* Device class */
static struct class* dev_class;


/* List of controller devices */
static struct list ctrl_list;


/* List of mapped host memory */
static struct list host_list;


/* List of mapped device memory */
static struct list device_list;


/* Number of devices */
static int max_num_ctrls = 64;
module_param(max_num_ctrls, int, 0);
MODULE_PARM_DESC(max_num_ctrls, "Number of controller devices");

static int curr_ctrls = 0;


static int mmap_registers(struct file* file, struct vm_area_struct* vma)
{
    struct ctrl* ctrl = NULL;
    int ret;

    /* Serialize against PCI remove which also holds map_create_mutex
     * before calling ctrl_put/kfree. This protects ctrl lifetime. */
    mutex_lock(&map_create_mutex);

    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (ctrl == NULL)
    {
        mutex_unlock(&map_create_mutex);
        printk(KERN_CRIT "Unknown controller reference\n");
        return -EBADF;
    }

    if (vma->vm_end - vma->vm_start > pci_resource_len(ctrl->pdev, 0))
    {
        mutex_unlock(&map_create_mutex);
        printk(KERN_WARNING "Invalid range size\n");
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    ret = vm_iomap_memory(vma, pci_resource_start(ctrl->pdev, 0), vma->vm_end - vma->vm_start);

    mutex_unlock(&map_create_mutex);
    return ret;
}



/*
 * Serialize on-the-fly map creation for the same (pid, vaddr).
 * Without this, two threads in the same process can race to create
 * duplicate kernel maps, and the second map_find_and_ref would miss
 * the first because it hasn't been list_insert-ed yet.
 */
DEFINE_MUTEX(map_create_mutex);

static long map_ioctl(struct file* file, unsigned int cmd, unsigned long arg)
{
    long retval = 0;
    struct ctrl* ctrl = NULL;
    struct nvm_ioctl_map request;
    struct map* map = NULL;
    u64 addr;

    /* Hold map_create_mutex for the entire ioctl to protect the
     * controller lifetime. remove_pci_dev also takes this mutex
     * before calling ctrl_put/kfree, so holding it guarantees
     * ctrl stays valid for the duration of this call. */
    mutex_lock(&map_create_mutex);

    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (ctrl == NULL)
    {
        mutex_unlock(&map_create_mutex);
        printk(KERN_CRIT "Unknown controller reference\n");
        return -EBADF;
    }

    switch (cmd)
    {
        case NVM_MAP_HOST_MEMORY:
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                retval = -EFAULT;
                break;
            }

            /* Reject n_pages==0 before it reaches map_find_and_ref
             * where it acts as a wildcard matcher. */
            if (request.n_pages == 0)
            {
                retval = -EINVAL;
                break;
            }

            map = map_find_and_ref(&host_list, request.vaddr_start, ctrl->pdev, request.n_pages, PAGE_SIZE);
            if (IS_ERR(map))
            {
                retval = PTR_ERR(map);
                break;
            }
            if (map != NULL)
            {
                /* Reuse existing mapping */
                if (copy_to_user((void __user*) request.ioaddrs, map->addrs,
                                  map->n_addrs * sizeof(uint64_t)))
                {
                    map_put_ref(map);
                    retval = -EFAULT;
                    break;
                }
                /* Recheck after copy_to_user: force_release may have
                 * fired during the copy. */
                if (atomic_read(&map->invalid))
                {
                    map_put_ref(map);
                    retval = -EIO;
                    break;
                }
                /* Success: keep the ref. It represents the second
                 * userspace handle's claim on this shared mapping.
                 * UNMAP will decrement via map_find_and_remove. */
                retval = 0;
                break;
            }

            /* Reject if an overlapping host mapping already exists
             * for this (pid, pdev). UNMAP uses wildcard matching
             * that ignores n_pages, so two host mappings at the
             * same vaddr with different sizes would be ambiguous. */
            if (map_range_exists(&host_list, request.vaddr_start,
                                 (u64)request.n_pages * PAGE_SIZE, ctrl->pdev))
            {
                retval = -EEXIST;
                break;
            }

            map = map_userspace(&host_list, ctrl, request.vaddr_start, request.n_pages);

            if (!IS_ERR_OR_NULL(map))
            {
                if (copy_to_user((void __user*) request.ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t)))
                {
                    unmap_and_release(map);
                    retval = -EFAULT;
                    break;
                }
                retval = 0;
            }
            else
            {
                retval = PTR_ERR(map);
            }
            break;

#ifdef _CUDA
        case NVM_MAP_DEVICE_MEMORY:
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                retval = -EFAULT;
                break;
            }

            /* Reject n_pages==0 before it reaches map_find_and_ref. */
            if (request.n_pages == 0)
            {
                retval = -EINVAL;
                break;
            }

            /* CUDA P2P maps use 64 KiB GPU pages */
            map = map_find_and_ref(&device_list, request.vaddr_start, ctrl->pdev, request.n_pages, 0x10000);
            if (IS_ERR(map))
            {
                retval = PTR_ERR(map);
                break;
            }
            if (map != NULL)
            {
                /* Reject if NVIDIA force-reclaimed the pages */
                if (atomic_read(&map->invalid) || map->data == NULL)
                {
                    map_put_ref(map);
                    retval = -EIO;
                    break;
                }
                if (copy_to_user((void __user*) request.ioaddrs, map->addrs,
                                  map->n_addrs * sizeof(uint64_t)))
                {
                    map_put_ref(map);
                    retval = -EFAULT;
                    break;
                }
                /* Recheck after copy_to_user: force_release may have
                 * fired during the copy (map_create_mutex does not
                 * serialize the NVIDIA callback). If so, the DMA
                 * addresses we just copied out are stale. */
                if (atomic_read(&map->invalid) || map->data == NULL)
                {
                    map_put_ref(map);
                    retval = -EIO;
                    break;
                }
                /* Success: keep the ref. It represents the second
                 * userspace handle's claim on this shared mapping.
                 * UNMAP will decrement via map_find_and_remove. */
                retval = 0;
                break;
            }

            /* Reject if a different backend (dmabuf) already mapped
             * an overlapping address range on this (pid, pdev).
             * UNMAP uses wildcard matching and cannot distinguish
             * backends, so allowing overlapping maps would let
             * UNMAP remove the wrong one. Uses range-overlap to
             * catch mappings at different page granularities.
             *
             * Use GPU-page-aligned base (same as map_device_memory
             * stores) so a request at B+0x1000 correctly detects
             * overlap with an existing map at B. */
            if (map_range_exists(&device_list, request.vaddr_start & ~(0x10000ULL - 1),
                                 (u64)request.n_pages * 0x10000ULL, ctrl->pdev))
            {
                retval = -EEXIST;
                break;
            }

            map = map_device_memory(&device_list, ctrl, request.vaddr_start, request.n_pages, &ctrl_list);

            if (!IS_ERR_OR_NULL(map))
            {
                /* Reject if force_release invalidated during setup */
                if (atomic_read(&map->invalid) || map->data == NULL)
                {
                    unmap_and_release(map);
                    retval = -EIO;
                    break;
                }
                if (copy_to_user((void __user*) request.ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t)))
                {
                    unmap_and_release(map);
                    retval = -EFAULT;
                    break;
                }
                /* Recheck after copy_to_user (see reuse path comment). */
                if (atomic_read(&map->invalid) || map->data == NULL)
                {
                    unmap_and_release(map);
                    retval = -EIO;
                    break;
                }
                retval = 0;
            }
            else
            {
                retval = PTR_ERR(map);
            }
            break;
#endif

#ifdef _HIP
        case NVM_MAP_DMABUF_MEMORY:
        {
            struct nvm_ioctl_dmabuf dreq;
            unsigned long n_pages;

            if (copy_from_user(&dreq, (void __user*) arg, sizeof(dreq)))
            {
                retval = -EFAULT;
                break;
            }

            /* Validate ioctl inputs */
            if (dreq.size == 0 || dreq.size % PAGE_SIZE != 0)
            {
                retval = -EINVAL;
                break;
            }
            if (dreq.gpu_ptr == 0 || dreq.gpu_ptr % PAGE_SIZE != 0)
            {
                retval = -EINVAL;
                break;
            }
            if (dreq.ioaddrs == 0 || dreq.ioaddrs_capacity == 0)
            {
                retval = -EINVAL;
                break;
            }
            /* Validate dmabuf_offset: must be page-aligned (or zero) */
            if (dreq.dmabuf_offset != 0 && dreq.dmabuf_offset % PAGE_SIZE != 0)
            {
                retval = -EINVAL;
                break;
            }

            n_pages = dreq.size / PAGE_SIZE;
            if (dreq.dmabuf_fd < 0)
            {
                retval = -EINVAL;
                break;
            }
            /* Sanity limit: reject absurdly large mappings */
            if (n_pages > 1024*1024) /* 4 GB max for 4 KiB pages */
            {
                retval = -EINVAL;
                break;
            }

            /* dmabuf reuse is disabled: the same gpu_ptr can be backed
             * by a different dmabuf_fd with different DMA addresses.
             * Each dmabuf map creates a new descriptor. */
            map = NULL;
            if (map != NULL)
            {
                /* Reuse existing mapping */
                if (atomic_read(&map->invalid))
                {
                    map_put_ref(map);
                    retval = -EIO;
                    break;
                }
                if (map->n_addrs > dreq.ioaddrs_capacity)
                {
                    map_put_ref(map);
                    retval = -EOVERFLOW;
                    break;
                }
                if (copy_to_user((void __user*)(uintptr_t)dreq.ioaddrs, map->addrs,
                                 map->n_addrs * sizeof(uint64_t)))
                {
                    map_put_ref(map);
                    retval = -EFAULT;
                    break;
                }
                /* Recheck after copy_to_user (dmabuf move_notify race). */
                if (atomic_read(&map->invalid))
                {
                    map_put_ref(map);
                    retval = -EIO;
                    break;
                }
                /* Success: keep the ref. It represents the second
                 * userspace handle's claim on this shared mapping.
                 * UNMAP will decrement via map_find_and_remove. */
                retval = 0;
                break;
            }

            /* Reject if a different backend (CUDA P2P) already mapped
             * an overlapping address range on this (pid, pdev).
             * UNMAP uses wildcard matching and cannot distinguish
             * backends, so allowing overlapping maps would let
             * UNMAP remove the wrong one. Uses range-overlap to
             * catch mappings at different page granularities. */
            if (map_range_exists(&device_list, dreq.gpu_ptr,
                                 (u64)n_pages * PAGE_SIZE, ctrl->pdev))
            {
                retval = -EEXIST;
                break;
            }

            map = map_dmabuf(&device_list, ctrl,
                             dreq.gpu_ptr,
                             dreq.dmabuf_fd,
                             dreq.dmabuf_offset,
                             n_pages,
                             dreq.ioaddrs_capacity);

            if (!IS_ERR_OR_NULL(map))
            {
                /* Fail-stop: reject if VRAM was invalidated during setup */
                if (atomic_read(&map->invalid))
                {
                    printk(KERN_ERR "uGDS: dmabuf mapping %llx invalidated "
                                    "during setup, pinned VRAM migration detected\n",
                           dreq.gpu_ptr);
                    unmap_and_release(map);
                    retval = -EIO;
                    break;
                }

                if (map->n_addrs > dreq.ioaddrs_capacity)
                {
                    unmap_and_release(map);
                    retval = -EOVERFLOW;
                    break;
                }
                if (copy_to_user((void __user*)(uintptr_t)dreq.ioaddrs, map->addrs,
                                 map->n_addrs * sizeof(uint64_t)))
                {
                    unmap_and_release(map);
                    retval = -EFAULT;
                    break;
                }
                /* Recheck after copy_to_user (dmabuf move_notify race). */
                if (atomic_read(&map->invalid))
                {
                    unmap_and_release(map);
                    retval = -EIO;
                    break;
                }
                retval = 0;
            }
            else
            {
                retval = PTR_ERR(map);
            }
            break;
        }
#endif

        case NVM_UNMAP_MEMORY:
            if (copy_from_user(&addr, (void __user*) arg, sizeof(u64)))
            {
                retval = -EFAULT;
                break;
            }

            {
                map = map_find_and_remove(&host_list, addr, ctrl->pdev);
                if (map == ERR_PTR(-EAGAIN))
                {
                    retval = 0;
                }
                else if (map != NULL)
                {
                    unmap_and_release(map);
                    retval = 0;
                }
                else
                {
#ifdef _CUDA
                    map = map_find_and_remove(&device_list, addr, ctrl->pdev);
                    if (map == ERR_PTR(-EAGAIN))
                    {
                        retval = 0;
                    }
                    else if (map != NULL)
                    {
                        unmap_and_release(map);
                        retval = 0;
                    }
                    else
#endif
                    {
#ifdef _HIP
                        map = map_find_and_remove(&device_list, addr, ctrl->pdev);
                        if (map == ERR_PTR(-EAGAIN))
                        {
                            retval = 0;
                        }
                        else if (map != NULL)
                        {
                            unmap_and_release(map);
                            retval = 0;
                        }
                        else
#endif
                        {
                            retval = -EINVAL;
                            printk(KERN_WARNING "Mapping for address %llx not found\n", addr);
                        }
                    }
                }
            }
            break;

        default:
            printk(KERN_NOTICE "Unknown ioctl command from process %d: %u\n",
                    current->pid, cmd);
            retval = -EINVAL;
            break;
    }

    mutex_unlock(&map_create_mutex);
    return retval;
}



/* Clean up all mappings owned by the closing file descriptor's process.
 * Called when userspace closes the device fd without explicit unmap. */
static int dev_release(struct inode* inode, struct file* file)
{
    struct pid* current_pid = get_task_pid(current, PIDTYPE_TGID);

    mutex_lock(&map_create_mutex);
    {
        struct list* lists[] = { &host_list, &device_list };
        int li;

        for (li = 0; li < 2; li++)
        {
            struct list_node* element;
            spin_lock(&lists[li]->lock);
            element = list_next(&lists[li]->head);
            while (element != NULL)
            {
                struct map* map = container_of(element, struct map, list);
                struct list_node* next = list_next(element);

                if (map->owner_pid == current_pid)
                {
                    /* Decrement refcount. If it reaches 0, unlink and
                     * free. If still referenced, leave in list for
                     * NVM_UNMAP_MEMORY to find later.
                     *
                     * After unlink (or simple decrement without removal)
                     * advance via 'next' -- NOT from the list head.
                     * Restarting from head would re-encounter the same
                     * entry whose refcount we just decremented, draining
                     * it to 0 and freeing shared mappings that are still
                     * in use. */
                    if (atomic_dec_and_test(&map->refcount))
                    {
                        element->prev->next = element->next;
                        element->next->prev = element->prev;
                        element->list = NULL;
                        element->next = NULL;
                        element->prev = NULL;

                        spin_unlock(&lists[li]->lock);
                        unmap_and_release(map);
                        spin_lock(&lists[li]->lock);
                    }

                    element = next;
                }
                else
                {
                    element = next;
                }
            }
            spin_unlock(&lists[li]->lock);
        }
    }
    mutex_unlock(&map_create_mutex);

    if (current_pid != NULL)
        put_pid(current_pid);

    return 0;
}


/* Define file operations for device file */
static const struct file_operations dev_fops =
{
    .owner = THIS_MODULE,
    .unlocked_ioctl = map_ioctl,
    .release = dev_release,
    .mmap = mmap_registers,
};


static int add_pci_dev(struct pci_dev* dev, const struct pci_device_id* id)
{
    int err;
    struct ctrl* ctrl = NULL;

    if (curr_ctrls >= max_num_ctrls)
    {
        printk(KERN_NOTICE "Maximum number of controller devices added\n");
        return 0;
    }

    printk(KERN_INFO "Adding controller device: %02x:%02x.%1x",
            dev->bus->number, PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn));

    // Create controller reference
    ctrl = ctrl_get(&ctrl_list, dev_class, dev, curr_ctrls);
    if (IS_ERR(ctrl))
    {
        return PTR_ERR(ctrl);
    }

    // Get a reference to device memory
    err = pci_request_region(dev, 0, DRIVER_NAME);
    if (err != 0)
    {
        ctrl_put(ctrl);
        printk(KERN_ERR "Failed to get controller register memory\n");
        return err;
    }

    // Enable PCI device
    err = pci_enable_device(dev);
    if (err < 0)
    {
        pci_release_region(dev, 0);
        ctrl_put(ctrl);
        printk(KERN_ERR "Failed to enable controller\n");
        return err;
    }

    // Create character device file
    err = ctrl_chrdev_create(ctrl, dev_first, &dev_fops);
    if (err != 0)
    {
        pci_disable_device(dev);
        pci_release_region(dev, 0);
        ctrl_put(ctrl);
        return err;
    }

    // Enable DMA
    pci_set_master(dev);

#if defined(_HIP)
    /* HIP backend requires 64-bit DMA for P2P VRAM addresses (large BAR).
     * 32-bit fallback is intentionally a hard failure -- AMD GPU P2P
     * DMA requires 64-bit addressing. */
    if (dma_set_mask_and_coherent(&dev->dev, DMA_BIT_MASK(64)))
    {
        printk(KERN_ERR DRIVER_NAME " HIP backend requires 64-bit DMA mask\n");
        pci_clear_master(dev);
        pci_disable_device(dev);
        pci_release_region(dev, 0);
        ctrl_put(ctrl);
        return -EIO;
    }
#else
    /* Default CUDA: try 64-bit, fall back to 32-bit. */
    if (dma_set_mask_and_coherent(&dev->dev, DMA_BIT_MASK(64)))
    {
        if (dma_set_mask_and_coherent(&dev->dev, DMA_BIT_MASK(32)))
        {
            printk(KERN_ERR DRIVER_NAME " failed to set DMA mask\n");
            pci_clear_master(dev);
            pci_disable_device(dev);
            pci_release_region(dev, 0);
            ctrl_put(ctrl);
            return -EIO;
        }
        printk(KERN_WARNING DRIVER_NAME " using 32-bit DMA mask\n");
    }
#endif

    /* Publish controller to the list only after probe is fully
     * complete. This prevents ioctls from using a partially
     * initialized controller. */
    ctrl_publish(&ctrl_list, ctrl);

    ++curr_ctrls;
    return 0;
}


static void remove_pci_dev(struct pci_dev* dev)
{
    struct ctrl* ctrl = NULL;
    printk(KERN_DEBUG DRIVER_NAME " Starting remove_pci_dev\n");
    if (dev == NULL)
    {
        printk(KERN_WARNING "Remove controller device was invoked with NULL\n");
        return;
    }

    /* Tear down all maps and the controller under map_create_mutex.
     * Holding the mutex through ctrl_put prevents new ioctls from
     * using the controller while it is being removed. */
    mutex_lock(&map_create_mutex);
    {
        struct list* lists[] = { &host_list, &device_list };
        int li;

        for (li = 0; li < 2; li++)
        {
            struct list_node* element;
            spin_lock(&lists[li]->lock);
            element = list_next(&lists[li]->head);
            while (element != NULL)
            {
                struct map* map = container_of(element, struct map, list);
                struct list_node* next = list_next(element);

                if (map->pdev == dev)
                {
                    element->prev->next = element->next;
                    element->next->prev = element->prev;
                    element->list = NULL;
                    element->next = NULL;
                    element->prev = NULL;

                    spin_unlock(&lists[li]->lock);
                    unmap_and_release(map);
                    spin_lock(&lists[li]->lock);

                    element = list_next(&lists[li]->head);
                }
                else
                {
                    element = next;
                }
            }
            spin_unlock(&lists[li]->lock);
        }

        /* Remove controller from list and destroy char device
         * while holding map_create_mutex so no new MAP ioctls
         * can start on this controller. */
        ctrl = ctrl_find_by_pci_dev(&ctrl_list, dev);
        if (ctrl != NULL)
            ctrl_put(ctrl);
    }
    mutex_unlock(&map_create_mutex);

    --curr_ctrls;

    // Release device memory
    pci_release_region(dev, 0);

    // Disable PCI device
    pci_clear_master(dev);
    pci_disable_device(dev);

    printk(KERN_DEBUG "Controller device removed: %02x:%02x.%1x\n",
            dev->bus->number, PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn));
}


static unsigned long clear_map_list(struct list* list)
{
    unsigned long i = 0;
    struct list_node* ptr;
    struct map* map;

    mutex_lock(&map_create_mutex);
    {
        ptr = list_next(&list->head);
        while (ptr != NULL)
        {
            map = container_of(ptr, struct map, list);
            unmap_and_release(map);
            ++i;

            ptr = list_next(&list->head);
        }
    }
    mutex_unlock(&map_create_mutex);

    return i;
}



/* Define driver operations we support */
static struct pci_driver driver = 
{
    .name = DRIVER_NAME,
    .id_table = id_table,
    .probe = add_pci_dev,
    .remove = remove_pci_dev,
};


static int __init ugds_drv_entry(void)
{
    int err;

    list_init(&ctrl_list);
    list_init(&host_list);
    list_init(&device_list);

    // Set up character device creation
    err = alloc_chrdev_region(&dev_first, 0, max_num_ctrls, DRIVER_NAME);
    if (err < 0)
    {
        printk(KERN_CRIT "Failed to allocate character device region\n");
        return err;
    }

    // Create character device class
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
    dev_class = class_create(DRIVER_NAME);
#else
    dev_class = class_create(THIS_MODULE, DRIVER_NAME);
#endif
    if (IS_ERR(dev_class))
    {
        unregister_chrdev_region(dev_first, max_num_ctrls);
        printk(KERN_CRIT "Failed to create character device class\n");
        return PTR_ERR(dev_class);
    }

    // Register as PCI driver
    err = pci_register_driver(&driver);
    if (err != 0)
    {
        class_destroy(dev_class);
        unregister_chrdev_region(dev_first, max_num_ctrls);
        printk(KERN_ERR "Failed to register as PCI driver\n");
        return err;
    }

    printk(KERN_DEBUG DRIVER_NAME " loaded\n");
    return 0;
}
module_init(ugds_drv_entry);


static void __exit ugds_drv_exit(void)
{
    unsigned long remaining = 0;

    remaining = clear_map_list(&device_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu GPU memory mappings were still in use on unload\n", remaining);
    }

    remaining = clear_map_list(&host_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu host memory mappings were still in use on unload\n", remaining);
    }
    printk(KERN_DEBUG DRIVER_NAME " Before pci_unregister_driver\n");
    pci_unregister_driver(&driver);
    printk(KERN_DEBUG DRIVER_NAME " After pci_unregister_driver\n");
    class_destroy(dev_class);
    unregister_chrdev_region(dev_first, max_num_ctrls);

    printk(KERN_DEBUG DRIVER_NAME " unloaded\n");
}
module_exit(ugds_drv_exit);
