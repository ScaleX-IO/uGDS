#include "ugds_internal.h"
#include "internal/dma.h"
#include <unistd.h>
#include <fcntl.h>
#include <new>

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
        uGDSBackend_t backend =
            nvm_dma_is_hip_origin(dma) ? UGDS_BACKEND_HIP : UGDS_BACKEND_CUDA;

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
