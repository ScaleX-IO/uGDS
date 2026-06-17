#include "ugds_internal.h"

extern "C" uGDSError_t uGDSBufRegister(const void* bufPtr_base, size_t length, int /*flags*/) {
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
    int status = nvm_dma_map_device(&dma, g_driver.default_ctrl,
                                    const_cast<void*>(bufPtr_base), length);
    if (status != 0 || dma == nullptr) {
        return make_error(UGDS_GPU_MEMORY_PINNING_FAILED);
    }

    g_driver.buf_registry[bufPtr_base] = dma;
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

    nvm_dma_unmap(it->second);
    g_driver.buf_registry.erase(it);

    return UGDS_OK;
}
