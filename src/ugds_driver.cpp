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

    /* Check ALL outstanding RDMA MRs before cleanup */
    for (const auto& pair : g_driver.rdma_records) {
        if (!pair.second.empty()) {
            return make_error(UGDS_RDMA_MR_STILL_ACTIVE);
        }
    }

    for (auto& entry : g_driver.buf_registry) {
        nvm_dma_unmap(entry.second.dma);
    }
    g_driver.buf_registry.clear();
    g_driver.rdma_records.clear();
    g_driver.default_ctrl = nullptr;
    g_driver.initialized = false;
    return UGDS_OK;
}
