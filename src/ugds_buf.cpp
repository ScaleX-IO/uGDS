#include "ugds_internal.h"
#include "internal/dma.h"
#include <unistd.h>
#include <fcntl.h>

extern "C" uGDSError_t uGDSBufRegister(const void* bufPtr_base, size_t length, int flags) {
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

    nvm_dma_t* dma = nullptr;
    int status = nvm_dma_map_device_ex(&dma, g_driver.default_ctrl,
                                       const_cast<void*>(bufPtr_base), length,
                                       flags);
    if (status != 0 || dma == nullptr) {
        if (status == ENOTSUP || status == EOPNOTSUPP)
            return make_error(UGDS_IO_NOT_SUPPORTED);
        return make_error(UGDS_GPU_MEMORY_PINNING_FAILED);
    }

    g_driver.buf_registry[bufPtr_base].dma = dma;
    g_driver.buf_registry[bufPtr_base].backend =
        nvm_dma_is_hip_origin(dma) ? UGDS_BACKEND_HIP : UGDS_BACKEND_CUDA;
    return UGDS_OK;
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
