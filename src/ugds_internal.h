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
#include <type_traits>

#define UGDS_DEFAULT_NUM_QPS     16
#define UGDS_DEFAULT_QUEUE_DEPTH 64
#define UGDS_BATCH_QUEUE_DEPTH   512
#define UGDS_MAX_BATCH_IO_SIZE   128
#define UGDS_PRP_POOL_PAGES      64
#define UGDS_HUGEPAGE_SIZE       (2UL * 1024 * 1024)

/* Fallback maximum data-transfer size (bytes) for a single I/O when the controller
 * reports MDTS = 0 (no limit). Keeps one transfer within a single PRP list. */
#define UGDS_DEFAULT_MAX_TRANSFER_SIZE (128UL * 1024)

/* SCT+SC (11-bit) from an NVMe completion: 0 = success. See NVMe Base Spec section 4.6.1. */
#define UGDS_CPL_SCT_SC(cpl)     (((cpl)->dword[3] >> 17) & 0x7FF)

/* Forward declarations: HandleOpGuard and SglRefOwner reference HandleState
 * and handle_release, which are fully defined later in this header. */
struct HandleState;
static inline void handle_release(HandleState* hs);

/* --- Per-IO resolved segment ------------------------------------------- */
/* A segment resolved against the buffer registry under one g_driver.lock
 * hold.  Identity fields (dma, base, registered_length, backend) are
 * immutable snapshots valid as long as the owning SglRefOwner holds its
 * reference; geometry fields (page_start, size) are filled at resolve time
 * (sync/batch) or in the async callback (late binding). */
struct SegView {
    nvm_dma_t*     dma;
    const void*    base;               /* registry key, for release/park */
    size_t         registered_length;  /* snapshot of BufEntry::length */
    uGDSBackend_t  backend;            /* snapshot, for async dispatch */
    size_t         page_start;         /* offset / MPS (offset MPS-aligned) */
    size_t         size;               /* bytes */
};

/* --- Reference ownership ----------------------------------------------- */
/* Owns one in_flight reference per SGL occurrence.  Duplicate bases are
 * intentionally repeated.  The representation has capacity for the public
 * maximum and never allocates; this is ~8 KiB on LP64 and is stored in
 * request/QP objects, not copied into a std::vector on timeout. */
class SglRefOwner {
    std::array<const void*, UGDS_IOV_MAX> bases_{};
    uint16_t size_ = 0;                 /* UGDS_IOV_MAX == 1024 */
public:
    SglRefOwner() noexcept = default;
    SglRefOwner(const SglRefOwner&) = delete;
    SglRefOwner& operator=(const SglRefOwner&) = delete;
    SglRefOwner(SglRefOwner&& other) noexcept;
    SglRefOwner& operator=(SglRefOwner&& other) noexcept;
    ~SglRefOwner() noexcept;            /* release() if still non-empty */

    bool empty() const noexcept { return size_ == 0; }
    uint16_t size() const noexcept { return size_; }
    const void* base_at(uint16_t i) const noexcept { return bases_[i]; }

    /* acquire(segs, nr): nr was already checked <= UGDS_IOV_MAX.  Under one
     * g_driver.lock hold, validate registration, exact length, and affinity,
     * then increment each counter and append its base.  All-or-nothing: a
     * failure rolls back 0..k-1 inside the same hold.  Returns 0 on success
     * or -EINVAL.  No reserve/growth/allocation exists.
     *
     * hs_ctrl is the submitting handle's controller (for affinity check).
     * page_size is MPS for the offset-alignment and page-index math.
     * seg_views_out (optional): if non-null, resolved SegView[] filled. */
    int acquire(const uGDSIoSegment_t* segs, uint32_t nr,
                const nvm_ctrl_t* hs_ctrl,
                size_t page_size,
                SegView* seg_views_out = nullptr);

    /* release(): decrement each stored occurrence under one g_driver.lock
     * hold and set size_ = 0.  Idempotent on empty. */
    void release() noexcept;
};

static_assert(std::is_nothrow_move_constructible<SglRefOwner>::value);
static_assert(std::is_nothrow_move_assignable<SglRefOwner>::value);
static_assert(UGDS_IOV_MAX <= UINT16_MAX);

/* --- Sync handle-operation ownership ----------------------------------- */
/* handle_lookup() increments the bare HandleState::handle_in_flight counter
 * independently of the returned shared_ptr.  This non-allocating guard is
 * constructed immediately after a successful lookup and must be declared
 * after the local hs_sp, so hs_sp outlives the guard.  Every early return
 * and exception therefore calls handle_release() exactly once.  The normal
 * path calls release() only after do_iov_engine returns; release() performs
 * handle_release() and then disarms the destructor. */
class HandleOpGuard {
    HandleState* hs_ = nullptr;
    bool         armed_ = false;
public:
    HandleOpGuard() noexcept = default;
    explicit HandleOpGuard(HandleState* hs) noexcept
        : hs_(hs), armed_(true) {}
    HandleOpGuard(const HandleOpGuard&) = delete;
    HandleOpGuard& operator=(const HandleOpGuard&) = delete;
    HandleOpGuard(HandleOpGuard&&) = delete;
    HandleOpGuard& operator=(HandleOpGuard&&) = delete;
    ~HandleOpGuard() noexcept { if (armed_) handle_release(hs_); }

    void arm(HandleState* hs) noexcept { hs_ = hs; armed_ = true; }
    void release() noexcept {
        if (armed_) { handle_release(hs_); armed_ = false; }
    }
};

static_assert(std::is_nothrow_destructible<HandleOpGuard>::value);

/* --- Engine result ------------------------------------------------------ */
/* An integer alone cannot drive cleanup: the caller must know whether the
 * engine consumed (parked) the resource owners. */
struct IovEngineResult {
    ssize_t ret;        /* bytes transferred or -errno */
    bool    timed_out;  /* true <=> handle wedged AND the engine moved the
                           owner (registered) or the transient dma
                           (on-the-fly) into QP timeout parking; the
                           caller's owner is empty / dma pointer stolen */
};

/* --- Command window ---------------------------------------------------- */
struct CmdWindow {
    uint32_t     first_seg;          /* index into SegView[] */
    uint32_t     n_segs;             /* slices spanned by this command */
    size_t       first_seg_page_off; /* pages already consumed from first_seg
                                        by earlier windows (mid-segment split) */
    size_t       bytes;              /* command length; block-multiple by construction */
    size_t       n_pages;            /* ceil-sum of slice pages; <= MPS/8 + 1 */
};

/* --- SglWindowCursor --------------------------------------------------- */
/* Stateful, allocation-free forward generator.  next() emits at most one
 * window and advances (seg_idx, bytes_in_seg, open-window state).  It never
 * owns or materializes an array of CmdWindow objects. */
struct SglWindowCursor {
    const SegView* segs;
    uint32_t       nr_segs;
    uint32_t       seg_idx;
    size_t         bytes_in_seg;     /* consumed so far in segs[seg_idx] */
    size_t         window_cap;
    size_t         page_size;        /* MPS */
    size_t         block_size;
    size_t         split_unit;       /* lcm(MPS, block_size) */
};

/* Initialize a cursor.  Returns false if window_cap == 0 (reject). */
bool sgl_cursor_init(SglWindowCursor& c, const SegView* segs, uint32_t nr_segs,
                     size_t window_cap, size_t page_size, size_t block_size);

/* Emit at most one window into out.  Returns true if a window was produced,
 * false when the SGL is exhausted. */
bool sgl_cursor_next(SglWindowCursor& c, CmdWindow& out) noexcept;

/* O(nr_segs) analytic count.  Produces the same count as the cursor. */
bool sgl_count_windows_analytic(const uGDSIoSegment_t* segs,
                                uint32_t nr_segs,
                                size_t window_cap, size_t page_size,
                                size_t block_size,
                                uint32_t* out) noexcept;

/* --- SglPageCursor ----------------------------------------------------- */
/* Forward iterator over the MPS page addresses of a window.  Replaces
 * buf_dma->ioaddrs[current_page + i] in the PRP builders.  Tracks position
 * across segment boundaries. */
struct SglPageCursor {
    const SegView* segs;
    uint32_t       seg_count;        /* n_segs from CmdWindow */
    uint32_t       seg_idx;          /* current segment index (relative to window) */
    size_t         page_in_seg;      /* absolute index into dma->ioaddrs */
    size_t         seg_pages_left;   /* pages remaining in current segment slice */
};

void sgl_page_cursor_init(SglPageCursor& c, const SegView* segs,
                          uint32_t first_seg, size_t first_seg_page_off,
                          uint32_t n_segs, size_t n_pages) noexcept;

/* Returns current ioaddr, advances across segment boundaries.  Total calls
 * bounded by CmdWindow::n_pages. */
uint64_t sgl_page_cursor_next(SglPageCursor& c) noexcept;

struct IOQueuePair {
    nvm_queue_t    sq{};
    nvm_queue_t    cq{};
    nvm_dma_t*     sq_dma  = nullptr;
    nvm_dma_t*     cq_dma  = nullptr;
    nvm_dma_t*     prp_dma = nullptr;
    void*          sq_buf  = nullptr;
    void*          cq_buf  = nullptr;
    void*          prp_buf = nullptr;
    int            irq_efd = -1;   /* eventfd for interrupt mode; -1 = poll */
    uint16_t       irq_vec = 0;    /* MSI-X vector bound to this CQ */
    /* Timeout resources remain owned by the QP until controller recovery.
     * At most one synchronous operation can hold this QP lock.
     * timeout_dma: on-the-fly (1-buf) mapping parked by the scalar engine.
     * timeout_refs: fixed-capacity registered-ref owner parked by the SGL
     *               engine (also used by the refactored scalar path). */
    nvm_dma_t*     timeout_dma = nullptr;           /* on-the-fly mapping */
    SglRefOwner    timeout_refs;                    /* registered refs */
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
    /* Owning recovery link to the active batch.  Together with
     * batch_active and batch_setting_up it forms one link-state machine
     * whose every transition happens inside a g_driver.lock critical
     * section.  Set by BatchIOSetUp publication; detached by exactly one
     * of the successful-destroy link transaction or the force-deregister
     * walk. */
    std::shared_ptr<struct BatchState> active_batch;
    /* True while a BatchIOSetUp that passed the setup gate (closing check
     * + batch_active CAS, one g_driver.lock section) has not yet committed
     * or rolled back.  Only the gate winner can set it; only that call
     * clears it (at publication, or as the LAST rollback action).  Written
     * only under g_driver.lock; read lock-free by force teardown, which
     * drains it to false before freeing any host-side QP/controller. */
    std::atomic<bool>           batch_setting_up{false};
    bool                        interrupt_mode = false;  /* UGDS_INTERRUPT_MODE */
    std::atomic<bool>           wedged{false};          /* batch timeout: handle poisoned, controller reset required */
    std::atomic<uint32_t>       handle_in_flight{0};  /* IO refcount for safe deregister */
    std::atomic<bool>           closing{false};        /* set by Deregister to block new ops */
};

struct DriverState {
    std::atomic<bool>                              initialized{false};
    std::mutex                                    lock;
    nvm_ctrl_t*                                   default_ctrl = nullptr;

    /* Handle registry: maps raw pointer -> shared_ptr to keep handles alive.
     * handle_lookup acquires under g_driver.lock so Deregister cannot
     * destroy the HandleState while an IO entrypoint is dereferencing it. */
    std::unordered_map<HandleState*,
                       std::shared_ptr<HandleState>>  handle_registry;

    /* Buffer registry with backend tracking for dual-backend dispatch.
     * in_flight counts active IO references to prevent use-after-free
     * during concurrent Deregister. */
    struct BufEntry {
        nvm_dma_t*           dma;
        uGDSBackend_t        backend;
        std::atomic<uint32_t> in_flight{0};
        size_t               length;    /* exact bytes from uGDSBufRegister */
        const nvm_ctrl_t*    map_ctrl;  /* controller at registration time */
        BufEntry() noexcept
            : dma(nullptr), backend(UGDS_BACKEND_DEFAULT),
              in_flight(0), length(0), map_ctrl(nullptr) {}
        BufEntry(nvm_dma_t* d, uGDSBackend_t b, size_t len,
                 const nvm_ctrl_t* ctrl) noexcept
            : dma(d), backend(b), in_flight(0), length(len), map_ctrl(ctrl) {}
    };
    std::unordered_map<const void*, BufEntry>     buf_registry;

    /* RDMA MR tracking */
    typedef enum {
        RDMA_REC_PENDING       = 0,   /* in-flight registration */
        RDMA_REC_ACTIVE        = 1,   /* MR registered, in use */
        RDMA_REC_DEREGISTERING = 2,   /* dereg in progress */
    } RDMARecordState;

    struct RDMARecord {
        const void*         bufPtr;
        void*               mr;         /* ibv_mr*, NULL while pending */
        int                 dup_fd;     /* -1 while pending */
        uint64_t            iova;
        uint64_t            offset;
        size_t              length;
        RDMARecordState     state;
        uint64_t            token;      /* unique per registration */
    };

    std::unordered_map<const void*, std::vector<RDMARecord>>  rdma_records;
    std::atomic<uint64_t>                                      rdma_token_counter{0};
};

extern DriverState g_driver;

/* Look up a handle in the global registry and acquire a reference.
 * Returns the raw HandleState* on success (and stores a shared_ptr copy
 * in *out_sp to keep the handle alive). Returns nullptr if the handle
 * is invalid or being deregistered.
 *
 * The caller MUST keep *out_sp alive for the duration of the operation
 * and call handle_release() when done. */
static inline HandleState* handle_lookup(uGDSHandle_t fh,
                                          std::shared_ptr<HandleState>* out_sp) {
    std::lock_guard<std::mutex> g(g_driver.lock);
    auto it = g_driver.handle_registry.find(static_cast<HandleState*>(fh));
    if (it == g_driver.handle_registry.end())
        return nullptr;
    if (it->second->closing.load(std::memory_order_acquire))
        return nullptr;
    if (it->second->wedged.load(std::memory_order_acquire))
        return nullptr;
    *out_sp = it->second;
    it->second->handle_in_flight.fetch_add(1, std::memory_order_acq_rel);
    return it->second.get();
}

/* Same as handle_lookup but assumes the caller already holds g_driver.lock.
 * Use to avoid deadlock when called from a context that already holds
 * the driver mutex (e.g. async_validate). */
static inline HandleState* handle_lookup_locked(uGDSHandle_t fh,
                                                  std::shared_ptr<HandleState>* out_sp) {
    auto it = g_driver.handle_registry.find(static_cast<HandleState*>(fh));
    if (it == g_driver.handle_registry.end())
        return nullptr;
    if (it->second->closing.load(std::memory_order_acquire))
        return nullptr;
    if (it->second->wedged.load(std::memory_order_acquire))
        return nullptr;
    *out_sp = it->second;
    it->second->handle_in_flight.fetch_add(1, std::memory_order_acq_rel);
    return it->second.get();
}

static inline void handle_release(HandleState* hs) {
    hs->handle_in_flight.fetch_sub(1, std::memory_order_acq_rel);
}

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

    /* SGL/lifecycle extensions */
    uint8_t             kind          = 0;  /* BatchEntryKind: 0=PLAIN, 1=VECTORED */
    uint32_t            seg_begin     = 0;  /* arena range start; valid iff VECTORED */
    uint32_t            seg_count     = 0;  /* number of segments in this entry */
    bool                refs_held     = false;  /* true while in-flight ref is held */
    bool                release_queued = false; /* true iff index is in release_scratch */
    uint16_t            n_cmds_submitted = 0;  /* successfully enqueued sub-commands */
};

/* Batch entry kind: plain (single buffer) or vectored (scatter-gather). */
enum BatchEntryKind : uint8_t {
    BATCH_ENTRY_PLAIN    = 0,
    BATCH_ENTRY_VECTORED = 1,
};

/* Batch lifecycle states (under bs->lock).
 * ACTIVE   -- normal operation; the public one-shot Destroy is available.
 * WEDGED   -- Destroy could not drain; one-shot Destroy was consumed.
 *             Force teardown drops self (no cycle retained).
 * TORN_DOWN-- all resources released; any further API call is a no-op. */
enum BatchLifecycle : uint8_t {
    BATCH_LIFECYCLE_ACTIVE    = 0,
    BATCH_LIFECYCLE_WEDGED    = 1,
    BATCH_LIFECYCLE_TORN_DOWN = 2,
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
    std::shared_ptr<HandleState> hs_sp;  /* keeps handle alive for batch lifetime */
    std::mutex   lock;

    /* Deferred-release scratch: entry indices whose registry refs must be
     * dropped after qp.lock is released.  Fixed-size, resized to capacity
     * at SetUp.  Written by index only under bs->lock; drained in one
     * g_driver.lock hold after qp.lock drops.  release_queued on each
     * BatchIOEntry is the authoritative membership bit. */
    std::vector<uint32_t> release_scratch;
    uint32_t              n_release_pending = 0;

    BatchLifecycle       lifecycle = BATCH_LIFECYCLE_ACTIVE;

    /* Public-handle ownership: the reference held on behalf of the user's
     * uGDSBatchHandle_t.  Accessed only under bs->lock.  Moved out -- never
     * reset in place -- at successful destroy, tombstone destroy, or
     * WEDGED-force cleanup, via the pin rule.  The object is destroyed when
     * the last reference (self, active_batch link, or transient pin) drops. */
    std::shared_ptr<BatchState> self;
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

/* --- SGL engine -------------------------------------------------------- */

/* Publish PRP list stores before the doorbell/data pointer.  Centralizes the
 * platform DMA-publish contract: on x86-64 the seq_cst fence is sufficient
 * because PRP lists live in cache-coherent host memory and normal stores
 * are ordered before the subsequent MMIO doorbell write.  Porting to a
 * weaker platform changes this one helper. */
static inline void sgl_publish_prp() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

/* Wait for one completion on a sync QP.  Returns nullptr on timeout.
 * Defined in ugds_io.cpp; shared with ugds_iov.cpp. */
nvm_cpl_t* wait_for_completion(HandleState* hs, IOQueuePair& qp);

/* Streaming windowed IO engine shared by scalar and vectored sync paths.
 *
 * hs: submitting handle state (must already be lookup-acquired).
 * segs: resolved segment views (identity + geometry complete).
 * nr_segs: number of entries in segs (>= 1).
 * file_offset: starting byte offset in the namespace (block-aligned).
 * opcode: NVM_IO_READ or NVM_IO_WRITE.
 * owner: holds in-flight buffer refs (may be empty if on_the_fly).
 * transient: on-the-fly DMA mapping, or nullptr for registered buffers.
 *
 * Returns {bytes, false} on success, {-errno, false} on device/validation
 * failure, or {-errno, true} on timeout with owner/transient moved into
 * qp.timeout_refs / qp.timeout_dma. */
IovEngineResult do_iov_engine(HandleState* hs, SegView* segs, uint32_t nr_segs,
                              off_t file_offset, uint8_t opcode,
                              SglRefOwner* owner, nvm_dma_t* transient);

struct AsyncRequest {
    uGDSHandle_t    fh;
    void*           bufPtr_base;
    size_t*         size_p;
    off_t*          file_offset_p;
    off_t*          bufPtr_offset_p;
    ssize_t*        bytes_done_p;
    uint8_t         opcode;
    std::shared_ptr<HandleState> hs_sp;  /* keeps handle alive until callback */
};

/* Internal stream type -- void* for backend neutrality */
typedef void* ugsd_stream_t;

void* hugepage_alloc(size_t size, size_t* alloc_size_out);
void  hugepage_free(void* ptr, size_t alloc_size);

#endif /* __UGDS_INTERNAL_H__ */
