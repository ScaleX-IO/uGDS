#ifndef __UGDS_INTERNAL_H__
#define __UGDS_INTERNAL_H__

#include "ugds.h"

#include <libnvm/nvm_types.h>
#include <libnvm/nvm_ctrl.h>
#include <libnvm/nvm_dma.h>
#include <libnvm/nvm_aq.h>
#include <libnvm/nvm_admin.h>
#include <libnvm/nvm_queue.h>
#include <libnvm/nvm_cmd.h>
#include <libnvm/nvm_util.h>
#include <libnvm/nvm_error.h>

#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

#define UGDS_DEFAULT_NUM_QPS     16
#define UGDS_DEFAULT_QUEUE_DEPTH 64

struct IOQueuePair {
    nvm_queue_t    sq;
    nvm_queue_t    cq;
    nvm_dma_t*     sq_dma;
    nvm_dma_t*     cq_dma;
    nvm_dma_t*     prp_dma;
    void*          sq_buf;
    void*          cq_buf;
    void*          prp_buf;
    std::mutex     lock;
};

struct HandleState {
    int                         fd;
    nvm_ctrl_t*                 ctrl;
    nvm_aq_ref                  aq_ref;
    nvm_dma_t*                  aq_dma;
    void*                       aq_buf;
    struct nvm_ctrl_info        ctrl_info;
    struct nvm_ns_info          ns_info;
    uint32_t                    ns_id;
    size_t                      block_size;
    size_t                      max_transfer_size;
    size_t                      max_transfer_pages;
    uint16_t                    num_qps;
    std::vector<std::unique_ptr<IOQueuePair>> qps;
    std::atomic<uint32_t>       rr_counter{0};
};

struct DriverState {
    bool                                          initialized = false;
    std::mutex                                    lock;
    nvm_ctrl_t*                                   default_ctrl = nullptr;
    std::unordered_map<const void*, nvm_dma_t*>   buf_registry;
};

extern DriverState g_driver;

static inline uGDSError_t make_error(uGDSOpError err) {
    uGDSError_t e;
    e.err = err;
    e.cu_err = 0;
    return e;
}

#define UGDS_OK make_error(UGDS_SUCCESS)

#endif /* __UGDS_INTERNAL_H__ */
