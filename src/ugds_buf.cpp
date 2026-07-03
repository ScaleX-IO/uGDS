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
        (flags & NVM_MAP_DMABUF) ? UGDS_BACKEND_HIP : UGDS_BACKEND_CUDA;
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
