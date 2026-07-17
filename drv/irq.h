/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef __UGDS_DRV_IRQ_H__
#define __UGDS_DRV_IRQ_H__

#include <linux/bitmap.h>
#include <linux/types.h>

#define UGDS_MAX_IRQ_VECTORS 64

struct ctrl;
struct pci_dev;

struct ugds_irq_state;

struct ugds_irq_owner
{
    DECLARE_BITMAP(owned, UGDS_MAX_IRQ_VECTORS);
};

static inline void ugds_irq_owner_init(struct ugds_irq_owner* o)
{
    bitmap_zero(o->owned, UGDS_MAX_IRQ_VECTORS);
}

void ugds_irq_ctrl_init(struct ctrl* ctrl, struct pci_dev* pdev);
void ugds_irq_ctrl_cleanup(struct ctrl* ctrl, struct pci_dev* pdev);
void ugds_irq_ctrl_destroy(struct ctrl* ctrl);

long ugds_irq_register(struct ctrl* ctrl, struct ugds_irq_owner* owner,
                       u32 vector, int eventfd);
long ugds_irq_unregister(struct ctrl* ctrl, struct ugds_irq_owner* owner,
                         u32 vector);
void ugds_irq_owner_cleanup(struct ctrl* ctrl, struct ugds_irq_owner* owner);

u32 ugds_irq_num_vectors(struct ctrl* ctrl);

#endif /* __UGDS_DRV_IRQ_H__ */
