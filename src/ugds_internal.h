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
#include <array>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

#define UGDS_DEFAULT_NUM_QPS     16
#define UGDS_DEFAULT_QUEUE_DEPTH 64
#define UGDS_BATCH_QUEUE_DEPTH   512
#define UGDS_MAX_BATCH_IO_SIZE   128
#define UGDS_PRP_POOL_PAGES      64
#define UGDS_HUGEPAGE_SIZE       (2UL * 1024 * 1024)

/* Fallback maximum data-transfer size (bytes) for a single I/O when the controller
 * reports MDTS = 0 (no limit). Keeps one transfer within a single PRP list. */
#define UGDS_DEFAULT_MAX_TRANSFER_SIZE (128UL * 1024)

/* SCT+SC (11-bit) from an NVMe completion: 0 = success. See NVMe Base Spec §4.6.1. */
#define UGDS_CPL_SCT_SC(cpl)     (((cpl)->dword[3] >> 17) & 0x7FF)

struct IOQueuePair {
    nvm_queue_t    sq{};
    nvm_queue_t    cq{};
    nvm_dma_t*     sq_dma  = nullptr;
    nvm_dma_t*     cq_dma  = nullptr;
    nvm_dma_t*     prp_dma = nullptr;
    void*          sq_buf  = nullptr;
    void*          cq_buf  = nullptr;
    void*          prp_buf = nullptr;
    std::mutex     lock;
};

struct IOQueuePairHuge {
    IOQueuePair                 qp;
    void*                       sq_huge      = nullptr;
    void*                       cq_huge      = nullptr;
    size_t                      sq_huge_size = 0;
    size_t                      cq_huge_size = 0;
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
    std::unique_ptr<IOQueuePairHuge> batch_qp;
    uint16_t                    batch_queue_depth;
    std::atomic<bool>           batch_active{false};
    std::atomic<uint32_t>       handle_in_flight{0};  /* async refcount for safe deregister */
};

struct DriverState {
    bool                                          initialized = false;
    std::mutex                                    lock;
    nvm_ctrl_t*                                   default_ctrl = nullptr;

    /* Buffer registry with backend tracking for dual-backend dispatch.
     * in_flight counts active IO references to prevent use-after-free
     * during concurrent Deregister. */
    struct BufEntry {
        nvm_dma_t*           dma;
        uGDSBackend_t        backend;
        std::atomic<uint32_t> in_flight{0};
    };
    std::unordered_map<const void*, BufEntry>     buf_registry;
};

extern DriverState g_driver;

struct PRPPool {
    nvm_dma_t*  dma       = nullptr;
    void*       buf       = nullptr;
    size_t      n_pages   = 0;
    uint64_t    free_bitmap = 0;
};

struct CmdSlot {
    uint16_t    io_idx      = 0;
    size_t      chunk_bytes = 0;
    uint16_t    prp_page_idx = UINT16_MAX;
    bool        active      = false;
};

struct BatchIOEntry {
    void*               cookie        = nullptr;
    void*               devPtr_base   = nullptr;
    off_t               file_offset   = 0;
    off_t               devPtr_offset = 0;
    size_t              size          = 0;
    uint8_t             opcode        = 0;

    uGDSBatchStatus_t   status        = UGDS_BATCH_WAITING;
    ssize_t             bytes_done    = 0;
    ssize_t             error_code    = 0;
    uint16_t            n_cmds        = 0;
    uint16_t            n_cmds_done   = 0;
    bool                event_returned = false;
};

struct BatchState {
    unsigned    capacity      = 0;
    unsigned    n_entries     = 0;
    unsigned    n_completed   = 0;
    unsigned    n_events_read = 0;

    std::vector<BatchIOEntry> entries;
    std::vector<CmdSlot>      cmd_map;
    uint16_t                  in_flight = 0;
    PRPPool                   prp_pool;

    HandleState* hs = nullptr;
    std::mutex   lock;
};

static inline uGDSError_t make_error(uGDSOpError err) {
    uGDSError_t e;
    e.err = err;
    e.cu_err = 0;
    return e;
}

#define UGDS_OK make_error(UGDS_SUCCESS)

ssize_t do_io_internal(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                       off_t file_offset, off_t bufPtr_offset, uint8_t opcode);

struct AsyncRequest {
    uGDSHandle_t    fh;
    void*           bufPtr_base;
    size_t*         size_p;
    off_t*          file_offset_p;
    off_t*          bufPtr_offset_p;
    ssize_t*        bytes_done_p;
    uint8_t         opcode;
};

/* Internal stream type — void* for backend neutrality */
typedef void* ugsd_stream_t;

void* hugepage_alloc(size_t size, size_t* alloc_size_out);
void  hugepage_free(void* ptr, size_t alloc_size);

#endif /* __UGDS_INTERNAL_H__ */
