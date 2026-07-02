#include "ugds_internal.h"

DriverState g_driver;

extern "C" uGDSError_t uGDSDriverOpen(void) {
    std::lock_guard<std::mutex> guard(g_driver.lock);
    if (g_driver.initialized) {
        return UGDS_OK;
    }
    g_driver.initialized = true;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSDriverClose(void) {
    std::lock_guard<std::mutex> guard(g_driver.lock);
    if (!g_driver.initialized) {
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    }

    /* Reject close if any buffer has outstanding IO.
     * Caller must ensure all async/batch operations are complete
     * and all buffers are deregistered before closing the driver. */
    for (auto& entry : g_driver.buf_registry) {
        if (entry.second.in_flight.load(std::memory_order_acquire) > 0) {
            return make_error(UGDS_BUSY);
        }
    }

    for (auto& entry : g_driver.buf_registry) {
        nvm_dma_unmap(entry.second.dma);
    }
    g_driver.buf_registry.clear();
    g_driver.default_ctrl = nullptr;
    g_driver.initialized = false;
    return UGDS_OK;
}
