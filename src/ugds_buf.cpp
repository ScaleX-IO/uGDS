#include "ugds_internal.h"
#include "internal/dma.h"
#include "internal/ioctl.h"
#include "internal/map.h"
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <new>
#include <cassert>

/* RAII guard for a device DMA mapping.
 * Arms nvm_dma_unmap on construction (or explicit arm()); disarm() must
 * be called after the mapping is successfully handed off to its owner
 * (e.g. inserted into the registry) so that the destructor does not
 * unmap it.  Used to make registry insertion transactional with respect
 * to bad_alloc. */
class MappedDmaGuard {
    nvm_dma_t* dma_ = nullptr;
    bool       armed_ = false;
public:
    MappedDmaGuard() noexcept = default;
    explicit MappedDmaGuard(nvm_dma_t* d) noexcept : dma_(d), armed_(true) {}
    ~MappedDmaGuard() {
        if (armed_ && dma_) nvm_dma_unmap(dma_);
    }
    MappedDmaGuard(const MappedDmaGuard&) = delete;
    MappedDmaGuard& operator=(const MappedDmaGuard&) = delete;
    MappedDmaGuard(MappedDmaGuard&&) = delete;
    MappedDmaGuard& operator=(MappedDmaGuard&&) = delete;

    void arm(nvm_dma_t* d) noexcept { dma_ = d; armed_ = true; }
    void disarm() noexcept { armed_ = false; }
    nvm_dma_t* get() const noexcept { return dma_; }
};

extern "C" uGDSError_t uGDSBufRegister(const void* bufPtr_base, size_t length, int flags) {
    try {
        if (!g_driver.initialized) {
            return make_error(UGDS_DRIVER_NOT_INITIALIZED);
        }

        if (bufPtr_base == nullptr || length == 0) {
            return make_error(UGDS_INVALID_VALUE);
        }

        std::lock_guard<std::mutex> guard(g_driver.lock);

        if (g_driver.buf_registry.find(bufPtr_base) != g_driver.buf_registry.end()) {
            return make_error(UGDS_MEMORY_ALREADY_REGISTERED);
        }

        if (g_driver.default_ctrl == nullptr) {
            return make_error(UGDS_DRIVER_NOT_INITIALIZED);
        }

        /* The library layer rejects a base that is not MPS-aligned,
         * that is not a multiple of the controller MPS. ioaddrs[] entries
         * are MPS-granular, so a non-MPS-aligned base means ioaddrs[0]
         * would not correspond to bufPtr_base. This is the universal PRP
         * requirement. */
        const size_t mps = g_driver.default_ctrl->page_size;
        if (mps > 0 &&
            (reinterpret_cast<uintptr_t>(bufPtr_base) % mps) != 0) {
            return make_error(UGDS_INVALID_VALUE);
        }

        nvm_dma_t* dma = nullptr;
        int status = nvm_dma_map_device_ex(&dma, g_driver.default_ctrl,
                                           const_cast<void*>(bufPtr_base), length,
                                           flags);
        if (status != 0 || dma == nullptr) {
            if (status == ENOTSUP || status == EOPNOTSUPP)
                return make_error(UGDS_IO_NOT_SUPPORTED);
            if (status == ENOMEM)
                return make_error(UGDS_OUT_OF_MEMORY);
            if (status == EINVAL)
                return make_error(UGDS_INVALID_VALUE);
            return make_error(UGDS_GPU_MEMORY_PINNING_FAILED);
        }

        /* Guard the mapping so a bad_alloc from registry insertion unmaps it
         * instead of leaking the device mapping across the C ABI.
         * The transactional insert below disarms the guard only after the
         * entry is fully constructed and stored. */
        MappedDmaGuard map_guard(dma);
        uGDSBackend_t backend;
        switch (nvm_dma_origin(dma)) {
        case NVM_DMABUF_ORIGIN_HIP:
            backend = UGDS_BACKEND_HIP;
            break;
        case NVM_DMABUF_ORIGIN_EXTERNAL:
            backend = UGDS_BACKEND_EXTERNAL;
            break;
        default:
            backend = UGDS_BACKEND_CUDA;
            break;
        }

        /* Post-map debug invariant: verify zero displacement so that
         * dma->ioaddrs[0] corresponds to the exact registered base.
         * A future mapping path that introduces displacement would
         * silently corrupt PRP construction. */
#ifndef NDEBUG
        if (dma->n_ioaddrs > 0) {
            assert((dma->ioaddrs[0] % mps) == 0);
        }
#endif

        /* try_emplace builds the BufEntry in place; if it throws bad_alloc
         * the mapping is still owned by map_guard and gets unmapped. */
        auto [it, inserted] = g_driver.buf_registry.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(bufPtr_base),
            std::forward_as_tuple(dma, backend, length, g_driver.default_ctrl));
        (void)it; (void)inserted;

        map_guard.disarm();
        return UGDS_OK;
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}

extern "C" uGDSError_t uGDSBufRegisterEx(const void* bufPtr_base, size_t length,
                                          const uGDSBufConfig_t* config) {
    if (!g_driver.initialized) {
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    }
    if (bufPtr_base == nullptr || length == 0 || config == nullptr) {
        return make_error(UGDS_INVALID_VALUE);
    }

    /* Build flags from config -- validate backend */
    int flags = 0;
    switch (config->backend) {
    case UGDS_BACKEND_DEFAULT:
        /* Export requires an explicit backend so the correct dma-buf
         * path is selected. DEFAULT relies on auto-probe which does
         * not retain an exportable fd. */
        if (config->enable_export)
            return make_error(UGDS_INVALID_VALUE);
        break;
    case UGDS_BACKEND_HIP:
#ifndef _HIP
        return make_error(UGDS_PLATFORM_NOT_SUPPORTED);
#else
        flags |= NVM_MAP_DMABUF;
        if (config->enable_export)
            flags |= NVM_MAP_RDMA;  /* retain dmabuf fd for export/RDMA */
#endif
        break;
    case UGDS_BACKEND_CUDA:
#ifndef _CUDA
        return make_error(UGDS_PLATFORM_NOT_SUPPORTED);
#else
        flags |= NVM_MAP_FORCE_CUDA;  /* skip auto-probe in dual-backend */
        if (config->enable_export)
            flags |= NVM_MAP_RDMA;    /* enable dmabuf export/RDMA path */
#endif
        break;
    default:
        return make_error(UGDS_INVALID_VALUE);
    }

    uGDSError_t st = uGDSBufRegister(bufPtr_base, length, flags);
    return st;
}

extern "C" uGDSError_t uGDSBufDeregister(const void* bufPtr_base) {
    if (!g_driver.initialized) {
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    }

    std::lock_guard<std::mutex> guard(g_driver.lock);

    auto it = g_driver.buf_registry.find(bufPtr_base);
    if (it == g_driver.buf_registry.end()) {
        return make_error(UGDS_MEMORY_NOT_REGISTERED);
    }

    /* Reject deregister if IO is in-flight on this buffer.
     * Caller must wait for outstanding operations to complete. */
    if (it->second.in_flight.load(std::memory_order_acquire) > 0) {
        return make_error(UGDS_BUSY);
    }

    /* Reject deregister if active RDMA MRs reference this buffer.
     * Caller must uGDSRDMAUnregister all MRs first.
     * DEREGISTERING state also blocks: ibv_dereg_mr may still be
     * in progress on another thread, and buffer unmap could race. */
    auto rdma_it = g_driver.rdma_records.find(bufPtr_base);
    if (rdma_it != g_driver.rdma_records.end()) {
        for (const auto& rec : rdma_it->second) {
            if (rec.state == DriverState::RDMA_REC_ACTIVE ||
                rec.state == DriverState::RDMA_REC_PENDING ||
                rec.state == DriverState::RDMA_REC_DEREGISTERING)
            {
                return make_error(UGDS_RDMA_MR_STILL_ACTIVE);
            }
        }
        /* All records are empty -- safe to clean up */
        g_driver.rdma_records.erase(rdma_it);
    }

    nvm_dma_unmap(it->second.dma);
    g_driver.buf_registry.erase(it);

    return UGDS_OK;
}

extern "C" uGDSError_t uGDSExportDmabuf(const void* bufPtr_base,
                                         uGDSDmabufExport_t* out) {
    if (!g_driver.initialized) {
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    }
    if (!out) {
        return make_error(UGDS_INVALID_VALUE);
    }

    std::lock_guard<std::mutex> guard(g_driver.lock);

    auto it = g_driver.buf_registry.find(bufPtr_base);
    if (it == g_driver.buf_registry.end()) {
        return make_error(UGDS_MEMORY_NOT_REGISTERED);
    }

    /* Get internal dmabuf metadata */
    int internal_fd = -1;
    uint64_t offset = 0;
    size_t length = 0;

    if (nvm_dma_get_dmabuf_info(it->second.dma, &internal_fd, &offset, &length) != 0
        || internal_fd < 0) {
        return make_error(UGDS_IO_NOT_SUPPORTED);
    }

    /* Atomically dup with CLOEXEC to prevent fd leaking into child
     * processes across fork/exec in multithreaded callers. */
    int dup_fd = fcntl(internal_fd, F_DUPFD_CLOEXEC, 0);
    if (dup_fd < 0) {
        return make_error(UGDS_INTERNAL_ERROR);
    }

    out->fd     = dup_fd;
    out->offset = offset;
    out->length = length;
    return UGDS_OK;
}


/* --- External dma-buf registration (Phase 6) ------------------------ */

/* Helper: translate a libnvm positive errno to a uGDS public error.
 * Follows the error translation table in the design doc section 13. */
static uGDSOpError translate_dmabuf_errno(int err)
{
    switch (err) {
    case EBADF:        return UGDS_BAD_FILE_DESCRIPTOR;
    case EINVAL:       return UGDS_INVALID_VALUE;
    case EMFILE:
    case ENFILE:       return UGDS_FD_LIMIT_REACHED;
    case ENOMEM:       return UGDS_OUT_OF_MEMORY;
    case EDQUOT:
    case ENOSPC:       return UGDS_PIN_LIMIT_EXCEEDED;
    case EBUSY:        return UGDS_BUSY;
    case EINTR:        return UGDS_INTERRUPTED;
    case ENODEV:       return UGDS_DEVICE_LOST;
    case EFAULT:       return UGDS_INVALID_USER_ADDRESS;
#ifdef EOPNOTSUPP
    case EOPNOTSUPP:   return UGDS_DMABUF_NOT_P2P;
#endif
#if defined(ENOTSUP) && (ENOTSUP != EOPNOTSUPP)
    case ENOTSUP:      return UGDS_IO_NOT_SUPPORTED;
#endif
    case ETIMEDOUT:    return UGDS_TIMED_OUT;
    case EOVERFLOW:    return UGDS_INVALID_MAPPING_SIZE;
    case ERANGE:       return UGDS_INVALID_MAPPING_RANGE;
    case EPROTONOSUPPORT:
    case ENOTTY:       return UGDS_STRUCT_VERSION_MISMATCH;
    case EPERM:
    case EACCES:       return UGDS_PERMISSION_DENIED;
    case EIO:          return UGDS_GPU_MEMORY_PINNING_FAILED;
    default:           return UGDS_INTERNAL_ERROR;
    }
}

extern "C" uGDSError_t uGDSBufRegisterDmabuf(const void* bufPtr_base,
                                               size_t length,
                                               const uGDSDmabufRegParams_t* params) {
#if !defined(UGDS_HAVE_DMABUF)
    (void)bufPtr_base; (void)length; (void)params;
    return make_error(UGDS_PLATFORM_NOT_SUPPORTED);
#else
    try {
        if (!g_driver.initialized) {
            return make_error(UGDS_DRIVER_NOT_INITIALIZED);
        }

        /* Validate params pointer */
        if (params == nullptr) {
            return make_error(UGDS_INVALID_VALUE);
        }

        /* Validate struct_size: must be at least the V1 size */
        if (params->struct_size < sizeof(uGDSDmabufRegParams_t)) {
            return make_error(UGDS_STRUCT_VERSION_MISMATCH);
        }

        /* Validate version */
        if (params->version != UGDS_DMABUF_REG_PARAMS_VERSION_1) {
            return make_error(UGDS_STRUCT_VERSION_MISMATCH);
        }

        /* Validate reserved fields are zero */
        if (params->reserved0 != 0) {
            return make_error(UGDS_INVALID_VALUE);
        }
        for (int i = 0; i < 4; i++) {
            if (params->reserved[i] != 0) {
                return make_error(UGDS_INVALID_VALUE);
            }
        }

        /* Validate no unknown flag bits */
        if (params->flags & ~((uint32_t)UGDS_DMABUF_REQUIRE_P2P)) {
            return make_error(UGDS_INVALID_VALUE);
        }

        /* Validate bufPtr_base and length */
        if (bufPtr_base == nullptr || length == 0) {
            return make_error(UGDS_INVALID_VALUE);
        }

        /* Validate dmabuf_fd */
        if (params->dmabuf_fd < 0) {
            return make_error(UGDS_BAD_FILE_DESCRIPTOR);
        }

        /* Host page size for alignment checks */
        long hps = sysconf(_SC_PAGESIZE);
        if (hps <= 0) {
            return make_error(UGDS_INTERNAL_ERROR);
        }

        /* Validate page alignment of bufPtr_base */
        if ((reinterpret_cast<uintptr_t>(bufPtr_base) % (size_t)hps) != 0) {
            return make_error(UGDS_INVALID_VALUE);
        }

        /* Validate page alignment of dmabuf_offset */
        if ((params->dmabuf_offset % (uint64_t)hps) != 0) {
            return make_error(UGDS_INVALID_VALUE);
        }

        /* MPS alignment (same check as uGDSBufRegister) */
        const size_t mps = g_driver.default_ctrl ? g_driver.default_ctrl->page_size : 0;
        if (mps > 0 && (reinterpret_cast<uintptr_t>(bufPtr_base) % mps) != 0) {
            return make_error(UGDS_INVALID_VALUE);
        }

        /* Lock the driver to check for duplicate registration and
         * capture the controller reference. */
        {
            std::lock_guard<std::mutex> guard(g_driver.lock);

            if (g_driver.buf_registry.find(bufPtr_base) != g_driver.buf_registry.end()) {
                return make_error(UGDS_MEMORY_ALREADY_REGISTERED);
            }

            if (g_driver.default_ctrl == nullptr) {
                return make_error(UGDS_DRIVER_NOT_INITIALIZED);
            }
        }

        /* The kernel ioctl is executed without holding g_driver.lock.
         * After the ioctl, we reacquire the lock and commit only if
         * the driver is still open and the key is still absent. */

        nvm_dma_t* dma = nullptr;
        int status;

        if (params->flags & UGDS_DMABUF_REQUIRE_P2P) {
            /* V2 path with strict P2P classification */
            struct nvm_ioctl_dmabuf_v2 v2result;
            memset(&v2result, 0, sizeof(v2result));

            status = nvm_dma_map_dmabuf_v2(&dma, g_driver.default_ctrl,
                                            const_cast<void*>(bufPtr_base),
                                            length,
                                            params->dmabuf_fd,
                                            params->dmabuf_offset,
                                            UGDS_DMABUF_REQUIRE_P2P,
                                            &v2result);

            /* Strict no-fallback: if REQUIRE_P2P was requested and the
             * kernel classified the mapping as not peer-BAR, do not
             * fall back to V1. */
            if (status == EOPNOTSUPP) {
                return make_error(UGDS_DMABUF_NOT_P2P);
            }
        } else {
            /* V1 path (no strict P2P requirement) */
            status = nvm_dma_map_dmabuf_fd(&dma, g_driver.default_ctrl,
                                            const_cast<void*>(bufPtr_base),
                                            length,
                                            params->dmabuf_fd,
                                            params->dmabuf_offset);
        }

        if (status != 0 || dma == nullptr) {
            return make_error(translate_dmabuf_errno(status));
        }

        /* Guard the mapping so a bad_alloc from registry insertion
         * unmaps it instead of leaking. */
        MappedDmaGuard map_guard(dma);

        /* Reacquire lock and commit */
        {
            std::lock_guard<std::mutex> guard(g_driver.lock);

            /* Re-check driver state and duplicate key */
            if (!g_driver.initialized) {
                return make_error(UGDS_DRIVER_CLOSING);
            }
            if (g_driver.buf_registry.find(bufPtr_base) != g_driver.buf_registry.end()) {
                return make_error(UGDS_MEMORY_ALREADY_REGISTERED);
            }

            auto [it, inserted] = g_driver.buf_registry.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(bufPtr_base),
                std::forward_as_tuple(dma, UGDS_BACKEND_EXTERNAL, length,
                                      g_driver.default_ctrl));
            (void)it; (void)inserted;
        }

        map_guard.disarm();
        return UGDS_OK;
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
#endif /* UGDS_HAVE_DMABUF */
}

extern "C" uGDSError_t uGDSQueryDmabufSupport(uGDSDmabufCaps_t* caps) {
    if (caps == nullptr) {
        return make_error(UGDS_INVALID_VALUE);
    }

    /* Zero-initialize the output */
    caps->lib_dmabuf = false;
    caps->kmod_dmabuf_v2 = false;
    caps->kmod_require_p2p = false;
    caps->kmod_pin_accounting = false;

#if !defined(UGDS_HAVE_DMABUF)
    /* Library compiled without dmabuf support */
    return UGDS_OK;
#else
    caps->lib_dmabuf = true;

    if (!g_driver.initialized || g_driver.default_ctrl == nullptr) {
        /* No controller: report library capability only */
        return UGDS_OK;
    }

    /* Issue NVM_GET_CAPS to the kernel */
    struct nvm_ioctl_caps kcaps;
    memset(&kcaps, 0, sizeof(kcaps));
    kcaps.struct_size = sizeof(kcaps);
    kcaps.version = NVM_CAPS_VERSION_1;

    int err = _nvm_dmabuf_get_caps(g_driver.default_ctrl, &kcaps);
    if (err != 0) {
        /* ENOTTY means the kernel does not support GET_CAPS (old module).
         * Report library capability only. */
        return UGDS_OK;
    }

    /* Parse kernel capability flags */
    if (kcaps.flags & NVM_CAP_DMABUF_V2) {
        caps->kmod_dmabuf_v2 = true;
    }
    if (kcaps.flags & NVM_CAP_DMABUF_REQUIRE_P2P) {
        caps->kmod_require_p2p = true;
    }
    if (kcaps.flags & NVM_CAP_DMABUF_PIN_ACCOUNTING) {
        caps->kmod_pin_accounting = true;
    }

    return UGDS_OK;
#endif /* UGDS_HAVE_DMABUF */
}
