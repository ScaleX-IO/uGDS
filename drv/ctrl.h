/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#ifndef __UGDS_DRV_CTRL_H__
#define __UGDS_DRV_CTRL_H__

#include "list.h"
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/kref.h>


/*
 * Represents an NVM controller.
 *
 * Lifetime is kref-managed. Holders:
 *   - ctrl_list holds one ref from probe (initial ref) until remove
 *   - every open file holds one ref from dev_open until dev_release
 * The embedded pdev is pinned with pci_dev_get for the ctrl lifetime.
 *
 * cdev is allocated separately with cdev_alloc(): inodes in the icache
 * hold a reference to the cdev kobject via i_cdev and can outlive the
 * last file AND the ctrl. An embedded cdev would be freed with the
 * ctrl while such inode references still exist (UAF on inode eviction).
 */
struct ctrl
{
    struct list_node    list;       /* Linked list head */
    struct kref         ref;        /* Lifetime refcount */
    struct pci_dev*     pdev;       /* Pinned PCI device (pci_dev_get) */
    char                name[64];   /* Character device name */
    int                 number;     /* Controller number */
    dev_t               rdev;       /* Character device register */
    struct class*       cls;        /* Character device class */
    struct cdev*        cdev;       /* Character device (cdev_alloc'd) */
    struct device*      chrdev;     /* Character device handle */
};



/*
 * Allocate a controller reference (initial kref = 1, pdev pinned).
 * Caller must ctrl_publish() after probe completes to make it visible.
 */
struct ctrl* ctrl_get(struct class* cls, struct pci_dev* pdev, int number);
void ctrl_publish(struct list* list, struct ctrl* ctrl);



/*
 * Drop a controller reference. When the last reference is dropped
 * the pdev is unpinned and the ctrl is freed.
 */
void ctrl_put(struct ctrl* ctrl);



/*
 * Find controller device (no reference taken; caller must already
 * hold one, e.g. the probe reference in remove_pci_dev).
 */
struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev);



/*
 * Find controller by inode and take a reference, atomically with
 * respect to remove (same list lock). Returns NULL if not found
 * or already dying.
 */
struct ctrl* ctrl_find_and_get_by_inode(struct list* list, const struct inode* inode);



/*
 * Create character device and set up file operations.
 */
int ctrl_chrdev_create(struct ctrl* ctrl,
                       dev_t first,
                       const struct file_operations* fops);



/*
 * Remove character device.
 */
void ctrl_chrdev_remove(struct ctrl* ctrl);



#endif /* __UGDS_DRV_CTRL_H__ */
