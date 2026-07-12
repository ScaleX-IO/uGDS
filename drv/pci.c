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
#include <linux/list.h>
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


/*
 * Index of open file contexts. This is NOT an ownership ledger --
 * every registration is owned by exactly one ugds_file_ctx. The
 * index exists solely so remove_pci_dev can find the files whose
 * registrations must be torn down when a controller goes away.
 *
 * Protected by ctx_list_mutex (a mutex, not a spinlock: the remove
 * path takes each ctx->lock, which sleeps, while iterating).
 * Lock order: ctx_list_mutex -> ctx->lock.
 */
static LIST_HEAD(ctx_list);
static DEFINE_MUTEX(ctx_list_mutex);


/* Number of devices */
static int max_num_ctrls = 64;
module_param(max_num_ctrls, int, 0);
MODULE_PARM_DESC(max_num_ctrls, "Number of controller devices");

static int curr_ctrls = 0;


enum handle_type
{
    HANDLE_HOST   = 1,
    HANDLE_CUDA   = 2,
    HANDLE_DMABUF = 3,
};


/*
 * Per-open-file registration ledger. One instance per struct file
 * (dup()/fork/fd-passing share the struct file and thus the ledger;
 * .release fires exactly once when the last reference closes).
 */
struct ugds_file_ctx
{
    struct list_head    global_node;    /* ctx_list membership */
    struct ctrl*        ctrl;           /* Resolved at open, kref held */
    struct list_head    handles;        /* map_handle ledger */
    struct mutex        lock;           /* Protects handles and dead */
    bool                dead;           /* Controller removed */
};


/*
 * One registration in a file's ledger. Identity is the full parameter
 * tuple: reuse requires all fields equal; a vaddr match with any other
 * field differing is -EEXIST (the UNMAP ABI only carries the address,
 * so two distinct handles at the same vaddr cannot coexist).
 */
struct map_handle
{
    struct list_head    node;
    struct map*         map;
    int                 type;           /* enum handle_type */
    u64                 vaddr;          /* Raw user address (MAP == UNMAP key) */
    unsigned long       n_pages;
    int                 dmabuf_fd;      /* -1 unless HANDLE_DMABUF */
    u64                 dmabuf_offset;
    unsigned int        count;          /* Registrations from this file */
};


static struct map_handle* handle_find(struct ugds_file_ctx* ctx, u64 vaddr)
{
    struct map_handle* handle;

    list_for_each_entry(handle, &ctx->handles, node)
    {
        if (handle->vaddr == vaddr)
        {
            return handle;
        }
    }

    return NULL;
}


static bool map_is_dead(int type, const struct map* map)
{
    if (atomic_read(&((struct map*) map)->invalid))
    {
        return true;
    }
#ifdef _CUDA
    /* NVIDIA force_release takes map->data via xchg */
    if (type == HANDLE_CUDA && map->data == NULL)
    {
        return true;
    }
#endif
    return false;
}


static int dev_open(struct inode* inode, struct file* file)
{
    struct ugds_file_ctx* ctx;
    struct ctrl* ctrl;

    ctrl = ctrl_find_and_get_by_inode(&ctrl_list, inode);
    if (ctrl == NULL)
    {
        printk(KERN_NOTICE "Open on unknown or removed controller\n");
        return -ENODEV;
    }

    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (ctx == NULL)
    {
        ctrl_put(ctrl);
        return -ENOMEM;
    }

    ctx->ctrl = ctrl;
    INIT_LIST_HEAD(&ctx->handles);
    mutex_init(&ctx->lock);
    ctx->dead = false;
    file->private_data = ctx;

    mutex_lock(&ctx_list_mutex);
    list_add(&ctx->global_node, &ctx_list);
    mutex_unlock(&ctx_list_mutex);

    return 0;
}


static int dev_release(struct inode* inode, struct file* file)
{
    struct ugds_file_ctx* ctx = file->private_data;
    struct map_handle* handle;
    struct map_handle* next;

    mutex_lock(&ctx_list_mutex);
    list_del(&ctx->global_node);
    mutex_unlock(&ctx_list_mutex);

    /* Each handle owns exactly one map regardless of count (count is
     * the number of registrations, not the number of maps). If the
     * controller was removed, the ledger is already empty. */
    mutex_lock(&ctx->lock);
    list_for_each_entry_safe(handle, next, &ctx->handles, node)
    {
        list_del(&handle->node);
        unmap_and_release(handle->map);
        kfree(handle);
    }
    mutex_unlock(&ctx->lock);

    ctrl_put(ctx->ctrl);
    kfree(ctx);

    return 0;
}


static int mmap_registers(struct file* file, struct vm_area_struct* vma)
{
    struct ugds_file_ctx* ctx = file->private_data;
    int ret;

    if (mutex_lock_killable(&ctx->lock))
    {
        return -EINTR;
    }

    if (ctx->dead)
    {
        mutex_unlock(&ctx->lock);
        return -ENODEV;
    }

    if (vma->vm_end - vma->vm_start > pci_resource_len(ctx->ctrl->pdev, 0))
    {
        mutex_unlock(&ctx->lock);
        printk(KERN_WARNING "Invalid range size\n");
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    ret = vm_iomap_memory(vma, pci_resource_start(ctx->ctrl->pdev, 0),
                          vma->vm_end - vma->vm_start);

    mutex_unlock(&ctx->lock);
    return ret;
}


static long do_map(struct ugds_file_ctx* ctx, enum handle_type type,
                   u64 vaddr, unsigned long n_pages,
                   int dmabuf_fd, u64 dmabuf_offset,
                   void __user* ioaddrs, size_t ioaddrs_capacity)
{
    struct map_handle* handle = NULL;
    struct map* map = NULL;
    long retval = 0;

    /* Per-file lock held across pinning and (bounded) fence waits:
     * only threads sharing this fd wait on each other. */
    if (mutex_lock_killable(&ctx->lock))
    {
        return -EINTR;
    }

    if (ctx->dead)
    {
        retval = -ENODEV;
        goto out;
    }

    handle = handle_find(ctx, vaddr);
    if (handle != NULL)
    {
        /* Reuse requires the full identity tuple to match. dmabuf
         * reuse is disabled entirely: the same gpu_ptr can be
         * re-exported as a different dmabuf with different DMA
         * addresses, so fd equality is not a robust identity. */
        if (type == HANDLE_DMABUF ||
            handle->type != (int) type ||
            handle->n_pages != n_pages ||
            handle->dmabuf_fd != dmabuf_fd ||
            handle->dmabuf_offset != dmabuf_offset)
        {
            retval = -EEXIST;
            goto out;
        }

        if (handle->count == UINT_MAX)
        {
            retval = -EBUSY;
            goto out;
        }

        if (map_is_dead(type, handle->map))
        {
            retval = -EIO;
            goto out;
        }

        handle->count++;
        if (copy_to_user(ioaddrs, handle->map->addrs,
                         handle->map->n_addrs * sizeof(uint64_t)))
        {
            handle->count--;
            retval = -EFAULT;
            goto out;
        }

        /* Recheck after copy_to_user: force_release may have fired
         * during the copy, making the copied addresses stale. */
        if (map_is_dead(type, handle->map))
        {
            handle->count--;
            retval = -EIO;
        }
        goto out;
    }

    switch (type)
    {
        case HANDLE_HOST:
            map = map_userspace(ctx->ctrl, vaddr, n_pages);
            break;

#ifdef _CUDA
        case HANDLE_CUDA:
            map = map_device_memory(ctx->ctrl, vaddr, n_pages, &ctrl_list);
            break;
#endif

#ifdef _HIP
        case HANDLE_DMABUF:
            map = map_dmabuf(ctx->ctrl, vaddr, dmabuf_fd, dmabuf_offset,
                             n_pages, ioaddrs_capacity);
            break;
#endif

        default:
            map = ERR_PTR(-EINVAL);
            break;
    }

    if (IS_ERR_OR_NULL(map))
    {
        retval = (map == NULL) ? -EIO : PTR_ERR(map);
        goto out;
    }

    if (map_is_dead(type, map))
    {
        if (type == HANDLE_DMABUF)
        {
            printk(KERN_ERR "uGDS: mapping %llx invalidated during setup, "
                            "pinned VRAM migration detected\n", vaddr);
        }
        unmap_and_release(map);
        retval = -EIO;
        goto out;
    }

    if (type == HANDLE_DMABUF && map->n_addrs > ioaddrs_capacity)
    {
        unmap_and_release(map);
        retval = -EOVERFLOW;
        goto out;
    }

    handle = kmalloc(sizeof(*handle), GFP_KERNEL);
    if (handle == NULL)
    {
        unmap_and_release(map);
        retval = -ENOMEM;
        goto out;
    }

    if (copy_to_user(ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t)))
    {
        kfree(handle);
        unmap_and_release(map);
        retval = -EFAULT;
        goto out;
    }

    /* Recheck after copy_to_user (see reuse path comment). Failure
     * here fully unwinds -- no ledger entry is left behind. */
    if (map_is_dead(type, map))
    {
        kfree(handle);
        unmap_and_release(map);
        retval = -EIO;
        goto out;
    }

    handle->map = map;
    handle->type = (int) type;
    handle->vaddr = vaddr;
    handle->n_pages = n_pages;
    handle->dmabuf_fd = dmabuf_fd;
    handle->dmabuf_offset = dmabuf_offset;
    handle->count = 1;
    list_add(&handle->node, &ctx->handles);

out:
    mutex_unlock(&ctx->lock);
    return retval;
}


static long do_unmap(struct ugds_file_ctx* ctx, u64 vaddr)
{
    struct map_handle* handle;
    long retval = 0;

    if (mutex_lock_killable(&ctx->lock))
    {
        return -EINTR;
    }

    if (ctx->dead)
    {
        retval = -ENODEV;
        goto out;
    }

    handle = handle_find(ctx, vaddr);
    if (handle == NULL)
    {
        printk(KERN_WARNING "Mapping for address %llx not found\n", vaddr);
        retval = -EINVAL;
        goto out;
    }

    if (--handle->count == 0)
    {
        list_del(&handle->node);
        unmap_and_release(handle->map);
        kfree(handle);
    }

out:
    mutex_unlock(&ctx->lock);
    return retval;
}


static long map_ioctl(struct file* file, unsigned int cmd, unsigned long arg)
{
    struct ugds_file_ctx* ctx = file->private_data;
    struct nvm_ioctl_map request;
    u64 addr;

    switch (cmd)
    {
        case NVM_MAP_HOST_MEMORY:
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                return -EFAULT;
            }
            if (request.n_pages == 0)
            {
                return -EINVAL;
            }
            return do_map(ctx, HANDLE_HOST, request.vaddr_start,
                          request.n_pages, -1, 0,
                          (void __user*) request.ioaddrs, SIZE_MAX);

#ifdef _CUDA
        case NVM_MAP_DEVICE_MEMORY:
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                return -EFAULT;
            }
            if (request.n_pages == 0)
            {
                return -EINVAL;
            }
            return do_map(ctx, HANDLE_CUDA, request.vaddr_start,
                          request.n_pages, -1, 0,
                          (void __user*) request.ioaddrs, SIZE_MAX);
#endif

#ifdef _HIP
        case NVM_MAP_DMABUF_MEMORY:
        {
            struct nvm_ioctl_dmabuf dreq;
            unsigned long n_pages;

            if (copy_from_user(&dreq, (void __user*) arg, sizeof(dreq)))
            {
                return -EFAULT;
            }

            /* Validate ioctl inputs */
            if (dreq.size == 0 || dreq.size % PAGE_SIZE != 0)
            {
                return -EINVAL;
            }
            if (dreq.gpu_ptr == 0 || dreq.gpu_ptr % PAGE_SIZE != 0)
            {
                return -EINVAL;
            }
            if (dreq.ioaddrs == 0 || dreq.ioaddrs_capacity == 0)
            {
                return -EINVAL;
            }
            /* Validate dmabuf_offset: must be page-aligned (or zero) */
            if (dreq.dmabuf_offset != 0 && dreq.dmabuf_offset % PAGE_SIZE != 0)
            {
                return -EINVAL;
            }
            if (dreq.dmabuf_fd < 0)
            {
                return -EINVAL;
            }

            n_pages = dreq.size / PAGE_SIZE;
            /* Sanity limit: reject absurdly large mappings */
            if (n_pages > 1024*1024) /* 4 GB max for 4 KiB pages */
            {
                return -EINVAL;
            }

            return do_map(ctx, HANDLE_DMABUF, dreq.gpu_ptr, n_pages,
                          dreq.dmabuf_fd, dreq.dmabuf_offset,
                          (void __user*)(uintptr_t) dreq.ioaddrs,
                          dreq.ioaddrs_capacity);
        }
#endif

        case NVM_UNMAP_MEMORY:
            if (copy_from_user(&addr, (void __user*) arg, sizeof(u64)))
            {
                return -EFAULT;
            }
            return do_unmap(ctx, addr);

        default:
            printk(KERN_NOTICE "Unknown ioctl command from process %d: %u\n",
                    current->pid, cmd);
            return -EINVAL;
    }
}


/* Define file operations for device file */
static const struct file_operations dev_fops =
{
    .owner = THIS_MODULE,
    .open = dev_open,
    .release = dev_release,
    .unlocked_ioctl = map_ioctl,
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
    ctrl = ctrl_get(dev_class, dev, curr_ctrls);
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
        ctrl_chrdev_remove(ctrl);
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
            ctrl_chrdev_remove(ctrl);
            pci_disable_device(dev);
            pci_release_region(dev, 0);
            ctrl_put(ctrl);
            return -EIO;
        }
        printk(KERN_WARNING DRIVER_NAME " using 32-bit DMA mask\n");
    }
#endif

    /* Publish controller to the list only after probe is fully
     * complete. This prevents opens from seeing a partially
     * initialized controller. */
    ctrl_publish(&ctrl_list, ctrl);

    ++curr_ctrls;
    return 0;
}


static void remove_pci_dev(struct pci_dev* dev)
{
    struct ctrl* ctrl = NULL;
    struct ugds_file_ctx* ctx;

    printk(KERN_DEBUG DRIVER_NAME " Starting remove_pci_dev\n");
    if (dev == NULL)
    {
        printk(KERN_WARNING "Remove controller device was invoked with NULL\n");
        return;
    }

    ctrl = ctrl_find_by_pci_dev(&ctrl_list, dev);
    if (ctrl == NULL)
    {
        /* Device was never adopted (probe capped by max_num_ctrls). */
        return;
    }

    /* 1. Unpublish: new opens can no longer find this controller.
     *    (list_remove takes the list spinlock -- same lock domain as
     *    ctrl_find_and_get_by_inode.) */
    list_remove(&ctrl->list);

    /* 2. Device node disappears. */
    ctrl_chrdev_remove(ctrl);

    /* 3. Settle accounts: for every open file on this controller,
     *    tear down its registrations and mark it dead. Acquiring
     *    ctx->lock means no MAP/UNMAP is in flight on that file;
     *    in-flight operations finish first (bounded: fence waits
     *    are capped at 10s). */
    mutex_lock(&ctx_list_mutex);
    list_for_each_entry(ctx, &ctx_list, global_node)
    {
        struct map_handle* handle;
        struct map_handle* next;

        if (ctx->ctrl != ctrl)
        {
            continue;
        }

        mutex_lock(&ctx->lock);
        list_for_each_entry_safe(handle, next, &ctx->handles, node)
        {
            list_del(&handle->node);
            unmap_and_release(handle->map);
            kfree(handle);
        }
        ctx->dead = true;
        mutex_unlock(&ctx->lock);
    }
    mutex_unlock(&ctx_list_mutex);

    /* 4. All DMA on this controller is now unmapped, satisfying the
     *    DMA API contract before the device goes away. */
    --curr_ctrls;
    pci_release_region(dev, 0);
    pci_clear_master(dev);
    pci_disable_device(dev);

    /* 5. Drop the probe reference. Open files still hold theirs;
     *    the ctrl is freed when the last one closes. */
    ctrl_put(ctrl);

    printk(KERN_DEBUG "Controller device removed: %02x:%02x.%1x\n",
            dev->bus->number, PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn));
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
    /* fops.owner pins the module while any file is open, so at this
     * point there are no live ctxs or maps from normal operation;
     * hot-remove/unbind teardown is handled by remove_pci_dev.
     * pci_unregister_driver invokes remove for every bound device. */
    printk(KERN_DEBUG DRIVER_NAME " Before pci_unregister_driver\n");
    pci_unregister_driver(&driver);
    printk(KERN_DEBUG DRIVER_NAME " After pci_unregister_driver\n");
    class_destroy(dev_class);
    unregister_chrdev_region(dev_first, max_num_ctrls);

    printk(KERN_DEBUG DRIVER_NAME " unloaded\n");
}
module_exit(ugds_drv_exit);
