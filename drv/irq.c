/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "irq.h"
#include "ctrl.h"
#include <linux/slab.h>
#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/eventfd.h>
#include <linux/version.h>
#include <linux/mutex.h>
#include <linux/errno.h>

#define DRIVER_NAME "ugds_drv"


struct ugds_irq_vec
{
    int                 irq;
    struct eventfd_ctx* efd;
    bool                requested;
};

/* The state outlives PCI removal so an in-flight ioctl can safely acquire
 * lock after cleanup has made the vector array unavailable. */
struct ugds_irq_state
{
    int                   num_vectors;
    struct ugds_irq_vec*  vecs;
    struct mutex          lock;
};


static irqreturn_t ugds_irq_handler(int irq, void* dev_id)
{
    struct ugds_irq_vec* v = dev_id;
    struct eventfd_ctx* efd = READ_ONCE(v->efd);

    if (efd != NULL)
    {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 8, 0)
        eventfd_signal(efd);
#else
        eventfd_signal(efd, 1);
#endif
    }

    return IRQ_HANDLED;
}


void ugds_irq_ctrl_init(struct ctrl* ctrl, struct pci_dev* pdev)
{
    struct ugds_irq_state* st;
    int want, nv, i;

    want = pci_msix_vec_count(pdev);
    if (want <= 0)
    {
        printk(KERN_INFO DRIVER_NAME " no MSI-X support; poll mode only\n");
        return;
    }
    if (want > UGDS_MAX_IRQ_VECTORS)
    {
        want = UGDS_MAX_IRQ_VECTORS;
    }

    st = kzalloc(sizeof(*st), GFP_KERNEL);
    if (st == NULL)
    {
        return;
    }
    mutex_init(&st->lock);

    st->vecs = kcalloc(want, sizeof(struct ugds_irq_vec), GFP_KERNEL);
    if (st->vecs == NULL)
    {
        kfree(st);
        return;
    }

    nv = pci_alloc_irq_vectors(pdev, 1, want, PCI_IRQ_MSIX);
    if (nv <= 0)
    {
        printk(KERN_WARNING DRIVER_NAME
               " pci_alloc_irq_vectors failed (%d); poll mode only\n", nv);
        kfree(st->vecs);
        kfree(st);
        return;
    }

    for (i = 0; i < nv; ++i)
    {
        st->vecs[i].irq = pci_irq_vector(pdev, i);
        st->vecs[i].efd = NULL;
        st->vecs[i].requested = false;
    }
    st->num_vectors = nv;
    ctrl->irq = st;

    printk(KERN_INFO DRIVER_NAME " allocated %d MSI-X vectors\n", nv);
}


void ugds_irq_ctrl_cleanup(struct ctrl* ctrl, struct pci_dev* pdev)
{
    struct ugds_irq_state* st = ctrl->irq;
    struct eventfd_ctx* pending[UGDS_MAX_IRQ_VECTORS];
    struct ugds_irq_vec* vecs;
    int i, n, pending_count = 0;

    if (st == NULL)
    {
        return;
    }

    mutex_lock(&st->lock);
    if (st->vecs == NULL)
    {
        mutex_unlock(&st->lock);
        return;
    }

    n = st->num_vectors;
    vecs = st->vecs;
    for (i = 0; i < n; ++i)
    {
        struct ugds_irq_vec* v = &vecs[i];
        struct eventfd_ctx* efd = v->efd;

        WRITE_ONCE(v->efd, NULL);
        if (v->requested)
        {
            free_irq(v->irq, v);
            v->requested = false;
        }
        if (efd != NULL)
            pending[pending_count++] = efd;
    }
    st->num_vectors = 0;
    st->vecs = NULL;
    mutex_unlock(&st->lock);

    for (i = 0; i < pending_count; ++i)
        eventfd_ctx_put(pending[i]);

    pci_free_irq_vectors(pdev);
    kfree(vecs);
}


void ugds_irq_ctrl_destroy(struct ctrl* ctrl)
{
    struct ugds_irq_state* st = ctrl->irq;

    if (st == NULL)
        return;

    WARN_ON(st->vecs != NULL);
    ctrl->irq = NULL;
    kfree(st);
}


long ugds_irq_register(struct ctrl* ctrl, struct ugds_irq_owner* owner,
                       u32 vector, int eventfd)
{
    struct ugds_irq_state* st = ctrl->irq;
    struct eventfd_ctx* efd;
    struct ugds_irq_vec* v;
    long retval = 0;

    if (vector >= UGDS_MAX_IRQ_VECTORS)
    {
        return -EINVAL;
    }

    efd = eventfd_ctx_fdget(eventfd);
    if (IS_ERR(efd))
    {
        return -EBADF;
    }

    if (st == NULL)
    {
        eventfd_ctx_put(efd);
        return -ENODEV;
    }

    mutex_lock(&st->lock);

    if (st->vecs == NULL || st->num_vectors == 0)
    {
        retval = -ENODEV;
        goto out;
    }
    if (vector >= (u32) st->num_vectors)
    {
        retval = -EINVAL;
        goto out;
    }

    v = &st->vecs[vector];
    if (v->efd != NULL)
    {
        retval = -EBUSY;
        goto out;
    }

    if (!v->requested)
    {
        retval = request_irq(v->irq, ugds_irq_handler, 0,
                             DRIVER_NAME, v);
        if (retval != 0)
        {
            goto out;
        }
        v->requested = true;
    }

    WRITE_ONCE(v->efd, efd);
    set_bit(vector, owner->owned);
    mutex_unlock(&st->lock);
    return 0;

out:
    mutex_unlock(&st->lock);
    eventfd_ctx_put(efd);
    return retval;
}


long ugds_irq_unregister(struct ctrl* ctrl, struct ugds_irq_owner* owner,
                         u32 vector)
{
    struct ugds_irq_state* st = ctrl->irq;
    struct ugds_irq_vec* v;
    struct eventfd_ctx* efd;

    if (vector >= UGDS_MAX_IRQ_VECTORS)
    {
        return -EINVAL;
    }

    if (st == NULL)
    {
        clear_bit(vector, owner->owned);
        return -ENODEV;
    }

    mutex_lock(&st->lock);

    if (st->vecs == NULL || st->num_vectors == 0)
    {
        clear_bit(vector, owner->owned);
        mutex_unlock(&st->lock);
        return -ENODEV;
    }
    if (vector >= (u32) st->num_vectors)
    {
        mutex_unlock(&st->lock);
        return -EINVAL;
    }

    if (!test_bit(vector, owner->owned))
    {
        mutex_unlock(&st->lock);
        return -EINVAL;
    }

    v = &st->vecs[vector];
    efd = v->efd;

    WRITE_ONCE(v->efd, NULL);
    if (v->requested)
    {
        free_irq(v->irq, v);
        v->requested = false;
    }
    clear_bit(vector, owner->owned);

    mutex_unlock(&st->lock);

    if (efd != NULL)
    {
        eventfd_ctx_put(efd);
    }

    return 0;
}


void ugds_irq_owner_cleanup(struct ctrl* ctrl, struct ugds_irq_owner* owner)
{
    unsigned long vec;

    for_each_set_bit(vec, owner->owned, UGDS_MAX_IRQ_VECTORS)
        ugds_irq_unregister(ctrl, owner, (u32) vec);
}


u32 ugds_irq_num_vectors(struct ctrl* ctrl)
{
    struct ugds_irq_state* st = ctrl->irq;
    u32 n;

    if (st == NULL)
        return 0;

    mutex_lock(&st->lock);
    n = (u32) st->num_vectors;
    mutex_unlock(&st->lock);

    return n;
}
