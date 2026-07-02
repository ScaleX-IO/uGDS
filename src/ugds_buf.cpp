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
        /* ENOTSUP or EINVAL when dmabuf flags are set indicates the
         * kernel module was not built with HAVE_CUDA_DMABUF and the
         * ioctl fell into its default case returning EINVAL. */
        if (status == ENOTSUP ||
            (status == EINVAL && (flags & NVM_MAP_DMABUF)))
            return make_error(UGDS_IO_NOT_SUPPORTED);
        return make_error(UGDS_GPU_MEMORY_PINNING_FAILED);
    }

    g_driver.buf_registry[bufPtr_base].dma = dma;
#if defined(_HIP) && defined(_CUDA)
    /* Dual-backend: detect actual backend from mapping origin.
     * nvm_dma_map_device_ex may route to HIP even with flags=0
     * (runtime pointer probing). Check hip_origin tag which persists
     * even after the dmabuf fd is closed post-kernel-import. */
    g_driver.buf_registry[bufPtr_base].backend =
        nvm_dma_is_hip_origin(dma) ? UGDS_BACKEND_HIP : UGDS_BACKEND_CUDA;
#else
    g_driver.buf_registry[bufPtr_base].backend =
        (flags & NVM_MAP_DMABUF) ? UGDS_BACKEND_HIP : UGDS_BACKEND_DEFAULT;
#endif
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

    /* Build flags from config — validate backend */
    int flags = 0;
    switch (config->backend) {
    case UGDS_BACKEND_DEFAULT:
        break;
    case UGDS_BACKEND_HIP:
#ifndef _HIP
        return make_error(UGDS_PLATFORM_NOT_SUPPORTED);
#else
        flags |= NVM_MAP_DMABUF;
        flags |= NVM_MAP_RDMA;  /* retain dmabuf fd for export */
#endif
        break;
    case UGDS_BACKEND_CUDA:
#ifndef _CUDA
        return make_error(UGDS_PLATFORM_NOT_SUPPORTED);
#else
        flags |= NVM_MAP_RDMA;  /* enable dmabuf export path */
#endif
        break;
    default:
        return make_error(UGDS_INVALID_VALUE);
    }

    uGDSError_t st = uGDSBufRegister(bufPtr_base, length, flags);
    if (st.err != UGDS_SUCCESS) return st;

    /* Store backend for runtime dispatch.
     * Skip overwrite when config->backend is DEFAULT — uGDSBufRegister
     * already detected the correct backend from the mapping origin.
     * Overwriting with DEFAULT would confuse dual-backend async dispatch. */
    if (config->backend != UGDS_BACKEND_DEFAULT) {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto it = g_driver.buf_registry.find(bufPtr_base);
        if (it != g_driver.buf_registry.end()) {
            it->second.backend = config->backend;
        }
    }
    return UGDS_OK;
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

    /* C-02 fix: dup() so caller owns the fd */
    int dup_fd = dup(internal_fd);
    if (dup_fd < 0) {
        return make_error(UGDS_INTERNAL_ERROR);
    }

    /* Set FD_CLOEXEC on the dup'd fd — fail hard on error (security) */
    int fd_flags = fcntl(dup_fd, F_GETFD);
    if (fd_flags < 0 || fcntl(dup_fd, F_SETFD, fd_flags | FD_CLOEXEC) < 0) {
        close(dup_fd);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    out->fd     = dup_fd;
    out->offset = offset;
    out->length = length;
    return UGDS_OK;
}
