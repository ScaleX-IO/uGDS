#include "ugds_internal.h"
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
        return make_error(UGDS_GPU_MEMORY_PINNING_FAILED);
    }

    g_driver.buf_registry[bufPtr_base] = dma;
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

    /* Build flags from config */
    int flags = 0;
    if (config->enable_rdma) {
        flags |= NVM_MAP_RDMA;
    }
    if (config->backend == UGDS_BACKEND_HIP) {
        flags |= NVM_MAP_DMABUF;
    }

    return uGDSBufRegister(bufPtr_base, length, flags);
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

    /* Check outstanding RDMA MRs before unmapping */
    auto mr_it = g_driver.rdma_records.find(bufPtr_base);
    if (mr_it != g_driver.rdma_records.end() && !mr_it->second.empty()) {
        return make_error(UGDS_RDMA_MR_STILL_ACTIVE);
    }

    nvm_dma_unmap(it->second);
    g_driver.buf_registry.erase(it);
    g_driver.rdma_records.erase(bufPtr_base);

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

    if (nvm_dma_get_dmabuf_info(it->second, &internal_fd, &offset, &length) != 0
        || internal_fd < 0) {
        return make_error(UGDS_IO_NOT_SUPPORTED);
    }

    /* C-02 fix: dup() so caller owns the fd */
    int dup_fd = dup(internal_fd);
    if (dup_fd < 0) {
        return make_error(UGDS_INTERNAL_ERROR);
    }

    /* Set FD_CLOEXEC on the dup'd fd */
    int fd_flags = fcntl(dup_fd, F_GETFD);
    if (fd_flags >= 0) {
        fcntl(dup_fd, F_SETFD, fd_flags | FD_CLOEXEC);
    }

    out->fd     = dup_fd;
    out->offset = offset;
    out->length = length;
    return UGDS_OK;
}
