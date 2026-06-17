/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#ifndef __UGDS_DRV_LIST_H__
#define __UGDS_DRV_LIST_H__

#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/compiler.h>


/* Forward declaration */
struct list;


/*
 * Doubly linked list element.
 */
struct list_node
{
    struct list*        list;   /* Reference to list */
    struct list_node*   next;   /* Pointer to next element in list */
    struct list_node*   prev;   /* Pointer to previous element in list */
};


/* 
 * Doubly linked list.
 * This implementation expects there always be an empty head.
 */
struct list
{
    struct list_node    head;   /* Start of the list */
    spinlock_t          lock;   /* Ensure exclusive access to list */
};



/*
 * Initialize element.
 */
static void __always_inline list_node_init(struct list_node* element)
{
    element->list = NULL;
    element->next = NULL;
    element->prev = NULL;
}



/*
 * Get next element in list (if there are any)
 */
#define list_next(current)  \
    ( ((current)->next != &(current)->list->head) ? (current)->next : NULL )



/*
 * Initialize list.
 */
void list_init(struct list* list);



/*
 * Insert element into list.
 */
void list_insert(struct list* list, struct list_node* element);



/*
 * Remove element from list.
 */
void list_remove(struct list_node* element);



#endif /* __UGDS_DRV_LIST_H__ */
