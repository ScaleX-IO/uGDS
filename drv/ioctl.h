/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markussen <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Originally derived from ssd-gpu-dma and BaM.
 */
#ifndef __UGDS_DRV_IOCTL_H__
#define __UGDS_DRV_IOCTL_H__
#ifdef __linux__

#include <linux/types.h>
#include <asm/ioctl.h>

#define NVM_IOCTL_TYPE          0x80

struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
};

/* Always define the dmabuf struct so the ioctl enum can reference it
 * unconditionally. The handler body is only compiled when
 * UGDS_HAVE_DMABUF is set, but the type must exist for the enum. */
struct nvm_ioctl_dmabuf
{
    __u64  gpu_ptr;           /* Original GPU VA -- unmap identity */
    __s32  dmabuf_fd;         /* DMA-buf fd from export */
    __u32  __pad;             /* Alignment */
    __u64  dmabuf_offset;     /* Offset within dmabuf allocation */
    __u64  size;              /* Total buffer size in bytes */
    __u64  ioaddrs_capacity;  /* Max entries in ioaddrs buffer */
    __u64  ioaddrs;           /* Output: DMA bus addresses (__user pointer) */
};

/* Bind/unbind an eventfd to an MSI-X interrupt vector (interrupt mode). */
struct nvm_ioctl_irq
{
    __u32  vector;            /* MSI-X vector index (0 .. num_vectors-1) */
    __s32  eventfd;           /* eventfd to signal on IRQ (register only) */
};

/* --- V2 dma-buf map ioctl -----------------------------------------
 *
 * Command 4 (NVM_MAP_DMABUF_MEMORY) is frozen for ABI compatibility.
 * V2 is command 9 with a versioned, extensible struct that adds:
 *   - UGDS_DMABUF_REQUIRE_P2P flag for strict peer-BAR enforcement
 *   - mapping_class output (SYSTEM vs PEER_BAR)
 *   - failure_reason for diagnostics
 *   - peer device identification (domain/bus/devfn/bar)
 *
 * The kernel classifies the mapped SG addresses against PCI BAR
 * windows of peer devices. When REQUIRE_P2P is set, any address
 * that cannot be matched to a single peer device's memory BAR
 * causes the mapping to fail with -EOPNOTSUPP before any DMA
 * address is exposed to userspace.
 */

#define NVM_DMABUF_MAP_VERSION_1      1u
#define UGDS_DMABUF_REQUIRE_P2P       (1u << 0)

enum nvm_dmabuf_mapping_class {
    NVM_DMABUF_MAPPING_UNKNOWN  = 0,
    NVM_DMABUF_MAPPING_SYSTEM   = 1,
    NVM_DMABUF_MAPPING_PEER_BAR = 2,
};

enum nvm_dmabuf_failure_reason {
    NVM_DMABUF_FAIL_NONE          = 0,
    NVM_DMABUF_FAIL_BAD_FD        = 1,
    NVM_DMABUF_FAIL_NOT_DMABUF    = 2,
    NVM_DMABUF_FAIL_RANGE         = 3,
    NVM_DMABUF_FAIL_PIN           = 4,
    NVM_DMABUF_FAIL_FENCE         = 5,
    NVM_DMABUF_FAIL_MAP           = 6,
    NVM_DMABUF_FAIL_SG_LAYOUT     = 7,
    NVM_DMABUF_FAIL_NOT_PEER_BAR  = 8,
    NVM_DMABUF_FAIL_PIN_LIMIT     = 9,
};

struct nvm_ioctl_dmabuf_v2 {
    __u32  struct_size;          /* sizeof(struct nvm_ioctl_dmabuf_v2) */
    __u16  version;              /* NVM_DMABUF_MAP_VERSION_1 */
    __u16  flags;                /* UGDS_DMABUF_REQUIRE_P2P, etc. */

    __u64  gpu_ptr;              /* Buffer VA -- ledger identity */
    __s32  dmabuf_fd;            /* DMA-buf fd from exporter */
    __u32  reserved0;            /* Must be zero */
    __u64  dmabuf_offset;        /* Offset within dma-buf object */
    __u64  size;                 /* Page-rounded map bytes */
    __u64  ioaddrs_capacity;     /* Max entries in ioaddrs buffer */
    __u64  ioaddrs;              /* Output: DMA bus addresses (__user) */

    /* Output fields -- valid when kernel can classify the attempt */
    __u32  mapping_class;        /* enum nvm_dmabuf_mapping_class */
    __u32  failure_reason;       /* enum nvm_dmabuf_failure_reason */
    __u16  peer_domain;          /* PCI domain of matched peer */
    __u8   peer_bus;             /* PCI bus number */
    __u8   peer_devfn;           /* PCI devfn */
    __u32  peer_bar;             /* BAR index on peer device */
    __u64  peer_bar_start;       /* Bus address of peer BAR */
    __u64  peer_bar_length;      /* Length of peer BAR */
    __u64  reserved[4];          /* Must be zero */
};

/* --- Capability query ioctl --------------------------------------- */

#define NVM_CAPS_VERSION_1            1u

#define NVM_CAP_DMABUF_V1             (1ull << 0)
#define NVM_CAP_DMABUF_V2             (1ull << 1)
#define NVM_CAP_DMABUF_REQUIRE_P2P    (1ull << 2)
#define NVM_CAP_DMABUF_MAPPING_CLASS  (1ull << 3)
#define NVM_CAP_DMABUF_PIN_ACCOUNTING (1ull << 4)

struct nvm_ioctl_caps {
    __u32  struct_size;                /* sizeof(struct nvm_ioctl_caps) */
    __u16  version;                    /* NVM_CAPS_VERSION_1 */
    __u16  kernel_uapi_version;        /* Highest V2 struct version understood */
    __u64  flags;                      /* NVM_CAP_* bitmask */
    __u64  max_dmabuf_map_bytes;       /* Per-map ceiling (1M pages * PAGE_SIZE) */
    __u64  per_file_pin_limit_bytes;   /* Module parameter: per-file pin limit */
    __u64  global_pin_limit_bytes;     /* Module parameter: global pin limit */
    __u64  current_file_pinned_bytes;  /* This file context's current pin charge */
    __u64  current_global_pinned_bytes;/* System-wide current pin charge */
    __u64  reserved[4];                /* Must be zero */
};

enum nvm_ioctl_type
{
    NVM_MAP_HOST_MEMORY         = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
#ifdef _CUDA
    NVM_MAP_DEVICE_MEMORY       = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
#endif
    NVM_UNMAP_MEMORY            = _IOW(NVM_IOCTL_TYPE, 3, uint64_t),
    NVM_MAP_DMABUF_MEMORY       = _IOWR(NVM_IOCTL_TYPE, 4, struct nvm_ioctl_dmabuf),
    NVM_REGISTER_INTERRUPT      = _IOW(NVM_IOCTL_TYPE, 5, struct nvm_ioctl_irq),
    NVM_UNREGISTER_INTERRUPT    = _IOW(NVM_IOCTL_TYPE, 6, struct nvm_ioctl_irq),
    NVM_GET_NUM_VECTORS         = _IOR(NVM_IOCTL_TYPE, 7, __u32),
    NVM_GET_CAPS                = _IOWR(NVM_IOCTL_TYPE, 8, struct nvm_ioctl_caps),
    NVM_MAP_DMABUF_MEMORY_V2    = _IOWR(NVM_IOCTL_TYPE, 9, struct nvm_ioctl_dmabuf_v2),
};

#endif /* __linux__ */
#endif /* __UGDS_DRV_IOCTL_H__ */
