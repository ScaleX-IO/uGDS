/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#include "ctrl.h"
#include "list.h"
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/kref.h>
#include <asm/errno.h>



static void ctrl_release(struct kref* ref)
{
    struct ctrl* ctrl = container_of(ref, struct ctrl, ref);

    pci_dev_put(ctrl->pdev);
    kfree(ctrl);
}



struct ctrl* ctrl_get(struct class* cls, struct pci_dev* pdev, int number)
{
    struct ctrl* ctrl = NULL;

    ctrl = kmalloc(sizeof(struct ctrl), GFP_KERNEL | GFP_NOWAIT);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Failed to allocate controller reference\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&ctrl->list);
    kref_init(&ctrl->ref);

    ctrl->pdev = pci_dev_get(pdev);
    ctrl->number = number;
    ctrl->rdev = 0;
    ctrl->cls = cls;
    ctrl->cdev = NULL;
    ctrl->chrdev = NULL;

    snprintf(ctrl->name, sizeof(ctrl->name), "%s%d", KBUILD_MODNAME, ctrl->number);
    ctrl->name[sizeof(ctrl->name) - 1] = '\0';

    /* NOTE: caller must call ctrl_publish() after probe completes
     * (DMA mask, chrdev, etc.) to make the controller visible. */
    return ctrl;
}

void ctrl_publish(struct list* list, struct ctrl* ctrl)
{
    list_insert(list, &ctrl->list);
}



void ctrl_put(struct ctrl* ctrl)
{
    if (ctrl != NULL)
    {
        kref_put(&ctrl->ref, ctrl_release);
    }
}



struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev)
{
    const struct list_node* element;
    struct ctrl* ctrl;

    spin_lock(&((struct list*)list)->lock);
    element = list_next(&list->head);
    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (ctrl->pdev == pdev)
        {
            spin_unlock(&((struct list*)list)->lock);
            return ctrl;
        }

        element = list_next(element);
    }
    spin_unlock(&((struct list*)list)->lock);

    return NULL;
}



struct ctrl* ctrl_find_and_get_by_inode(struct list* list, const struct inode* inode)
{
    const struct list_node* element;
    struct ctrl* ctrl;

    spin_lock(&list->lock);
    element = list_next(&list->head);
    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (ctrl->cdev == inode->i_cdev)
        {
            /* Same lock domain as list removal in remove_pci_dev,
             * so a controller found here cannot have dropped its
             * final reference yet. */
            if (!kref_get_unless_zero(&ctrl->ref))
            {
                break;
            }
            spin_unlock(&list->lock);
            return ctrl;
        }

        element = list_next(element);
    }
    spin_unlock(&list->lock);

    return NULL;
}



int ctrl_chrdev_create(struct ctrl* ctrl, dev_t first, const struct file_operations* fops)
{
    int err;
    struct device* chrdev = NULL;

    if (ctrl->chrdev != NULL)
    {
        printk(KERN_WARNING "Character device is already created\n");
        return 0;
    }

    ctrl->rdev = MKDEV(MAJOR(first), MINOR(first) + ctrl->number);

    ctrl->cdev = cdev_alloc();
    if (ctrl->cdev == NULL)
    {
        printk(KERN_ERR "Failed to allocate cdev\n");
        return -ENOMEM;
    }
    ctrl->cdev->owner = fops->owner;
    ctrl->cdev->ops = fops;

    err = cdev_add(ctrl->cdev, ctrl->rdev, 1);
    if (err != 0)
    {
        kobject_put(&ctrl->cdev->kobj);
        ctrl->cdev = NULL;
        printk(KERN_ERR "Failed to add cdev\n");
        return err;
    }

    chrdev = device_create(ctrl->cls, NULL, ctrl->rdev, NULL, ctrl->name);
    if (IS_ERR(chrdev))
    {
        cdev_del(ctrl->cdev);
        ctrl->cdev = NULL;
        printk(KERN_ERR "Failed to create character device\n");
        return PTR_ERR(chrdev);
    }

    ctrl->chrdev = chrdev;

    printk(KERN_INFO "Character device /dev/%s created (%d.%d)\n",
            ctrl->name, MAJOR(ctrl->rdev), MINOR(ctrl->rdev));

    return 0;
}



void ctrl_chrdev_remove(struct ctrl* ctrl)
{
    if (ctrl->chrdev != NULL)
    {
        device_destroy(ctrl->cls, ctrl->rdev);
        cdev_del(ctrl->cdev);
        ctrl->cdev = NULL;
        ctrl->chrdev = NULL;

        printk(KERN_DEBUG "Character device /dev/%s removed (%d.%d)\n",
                ctrl->name, MAJOR(ctrl->rdev), MINOR(ctrl->rdev));
    }
}
