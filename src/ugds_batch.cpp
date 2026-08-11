#include "ugds_internal.h"

#include <cstring>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <atomic>
#include <time.h>
#include <cstdio>
#include <cassert>
#include <new>
#include <memory>

/* Release in-flight reference held by batch submit.
 * Called on validation failure or when batch entry completes.
 * MUST be called WITHOUT holding g_driver.lock. */
static void async_release_inflight_batch(void* devPtr_base)
{
    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    auto it = g_driver.buf_registry.find(devPtr_base);
    if (it != g_driver.buf_registry.end())
        it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
}

/* Append an entry index to the deferred-release scratch.
 * Must be called under bs->lock; the actual registry decrement is
 * deferred until after qp.lock is released. */
static inline void queue_entry_release(BatchState* bs, unsigned idx)
{
    BatchIOEntry& entry = bs->entries[idx];
    if (!entry.refs_held) return;
    if (entry.release_queued) return;  /* already queued */
    assert(bs->n_release_pending < bs->release_scratch.size());
    entry.release_queued = true;
    bs->release_scratch[bs->n_release_pending++] = idx;
}

/* Drain all pending deferred releases in one g_driver.lock hold.
 * Must be called under bs->lock but NOT under qp.lock and NOT under
 * g_driver.lock (the old version called
 * async_release_inflight_batch which re-locked g_driver.lock, causing
 * a deadlock on a non-recursive mutex). */
static void drain_release_scratch(BatchState* bs)
{
    if (bs->n_release_pending == 0) return;
    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    for (uint32_t i = 0; i < bs->n_release_pending; ++i) {
        uint32_t idx = bs->release_scratch[i];
        BatchIOEntry& entry = bs->entries[idx];
        if (!entry.refs_held) continue;  /* already released */
        /* Inline the decrement instead of calling
         * async_release_inflight_batch to avoid re-locking
         * g_driver.lock (nested-lock bug fix). */
        auto it = g_driver.buf_registry.find(entry.devPtr_base);
        if (it != g_driver.buf_registry.end())
            it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
        entry.refs_held = false;
        entry.release_queued = false;
    }
    bs->n_release_pending = 0;
}

/* Abort a Phase-3 entry that is WAITING or PENDING.
 * Truncates n_cmds to the submitted prefix (n_cmds_submitted), records
 * -ECANCELED, and queues the entry for deferred release if all its
 * submitted commands have already completed.  Must be called
 * under bs->lock + qp.lock; does NOT touch g_driver.lock (R2 MINOR-1). */
static inline void abort_phase3_entry(BatchState* bs, unsigned idx)
{
    BatchIOEntry& entry = bs->entries[idx];
    if (entry.status != UGDS_BATCH_WAITING &&
        entry.status != UGDS_BATCH_PENDING)
        return;
    entry.error_code = -ECANCELED;
    entry.n_cmds = entry.n_cmds_submitted;  /* discard unsubmitted suffix */
    if (entry.n_cmds_done == entry.n_cmds) {
        /* All submitted commands already drained: terminalize now. */
        entry.status = UGDS_BATCH_FAILED;
        bs->n_completed++;
        queue_entry_release(bs, idx);
    }
    /* else: submitted prefix still draining; drain_one_completion will
     * terminalize when n_cmds_done == n_cmds. */
}

void cleanup_prp_pool(BatchState* bs)
{
    PRPPool& pool = bs->prp_pool;
    if (pool.dma) nvm_dma_unmap(pool.dma);
    free(pool.buf);
    pool.dma = nullptr;
    pool.buf = nullptr;
}

static int prp_pool_alloc(PRPPool* pool)
{
    if (pool->free_bitmap == 0) return -1;
    int idx = __builtin_ctzll(pool->free_bitmap);
    pool->free_bitmap &= ~(1ULL << idx);
    return idx;
}

static void prp_pool_free(PRPPool* pool, int idx)
{
    pool->free_bitmap |= (1ULL << idx);
}

static bool drain_one_completion(IOQueuePair& qp, BatchState* bs)
{
    nvm_cpl_t* cpl = nvm_cq_dequeue(&qp.cq);
    if (!cpl) return false;

    uint16_t cid = *NVM_CPL_CID(cpl);
    uint16_t status = UGDS_CPL_SCT_SC(cpl);

    nvm_sq_update(&qp.sq);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    nvm_cq_update(&qp.cq);

    CmdSlot& slot = bs->cmd_map[cid];
    BatchIOEntry& entry = bs->entries[slot.io_idx];

    if (status != 0) {
        entry.status = UGDS_BATCH_FAILED;
    } else {
        entry.bytes_done += slot.chunk_bytes;
    }

    if (slot.prp_page_idx != UINT16_MAX) {
        prp_pool_free(&bs->prp_pool, slot.prp_page_idx);
    }
    slot.active = false;
    bs->in_flight--;

    entry.n_cmds_done++;

    if (entry.error_code == -ECANCELED) {
        /* Abort has terminal-result precedence: do not overwrite with
         * bytes or device status.  Terminalize when the submitted prefix
         * has fully drained. */
        if (entry.n_cmds_done == entry.n_cmds) {
            entry.status = UGDS_BATCH_FAILED;
            bs->n_completed++;
            queue_entry_release(bs, slot.io_idx);
        }
    } else if (entry.n_cmds_done == entry.n_cmds) {
        if (entry.status != UGDS_BATCH_FAILED) {
            entry.status = UGDS_BATCH_COMPLETE;
        }
        entry.error_code = entry.bytes_done;
        bs->n_completed++;
        /* Queue deferred release of in-flight reference:
         * the actual registry decrement runs after qp.lock drops via
         * drain_release_scratch, restoring the stated lock order. */
        queue_entry_release(bs, slot.io_idx);
    }

    return true;
}

static size_t compute_max_xfer(HandleState* hs)
{
    const size_t page_size = hs->ctrl->page_size;
    const size_t prp_capacity = page_size / sizeof(uint64_t);

    size_t max_xfer = hs->max_transfer_size;
    if (max_xfer == 0) max_xfer = UGDS_DEFAULT_MAX_TRANSFER_SIZE;
    if (max_xfer < page_size) max_xfer = page_size;

    size_t prp_max = (prp_capacity + 1) * page_size;
    if (max_xfer > prp_max) max_xfer = prp_max;

    return max_xfer;
}

extern "C" uGDSError_t uGDSBatchIOSetUp(uGDSBatchHandle_t* batch,
                                          uGDSHandle_t fh, unsigned nr)
{
    try {
    if (batch == nullptr || fh == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (nr == 0 || nr > UGDS_MAX_BATCH_IO_SIZE)
        return make_error(UGDS_INVALID_VALUE);

    /* --- Setup gate: one g_driver.lock critical section ---
     * The closing check, batch_active claim, and batch_setting_up
     * publication are indivisible with respect to force teardown's
     * closing claim, which runs under the same lock. */
    std::shared_ptr<HandleState> hs_sp;
    HandleState* hs = nullptr;
    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        auto it = g_driver.handle_registry.find(static_cast<HandleState*>(fh));
        if (it == g_driver.handle_registry.end())
            return make_error(UGDS_INVALID_VALUE);
        if (it->second->closing.load(std::memory_order_acquire))
            return make_error(UGDS_INVALID_VALUE);
        if (it->second->wedged.load(std::memory_order_acquire))
            return make_error(UGDS_INVALID_VALUE);
        if (!it->second->batch_qp)
            return make_error(UGDS_INTERNAL_ERROR);

        bool expected = false;
        if (!it->second->batch_active.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel))
            return make_error(UGDS_INVALID_VALUE);

        /* Gate won: acquire handle reference + set batch_setting_up */
        hs_sp = it->second;
        it->second->handle_in_flight.fetch_add(1, std::memory_order_acq_rel);
        hs = hs_sp.get();
        hs->batch_setting_up.store(true, std::memory_order_release);
    }

    /* M3: BatchSetupTxn - RAII scope guard that rolls back gate state
     * if the function exits via exception or early return before commit.
     * Five items (acquired in order, rolled back in reverse):
     *   1. handle_in_flight reference
     *   2. batch_active claim
     *   3. batch_setting_up flag (cleared LAST, under g_driver.lock)
     * Items 4-5 (pool DMA, pool buf) are handled by the explicit
     * setup_rollback label for non-throwing failures. */
    struct BatchSetupTxn {
        HandleState* hs;
        bool armed;
        ~BatchSetupTxn() {
            if (!armed) return;
            std::lock_guard<std::mutex> g(g_driver.lock);
            hs->batch_active.store(false, std::memory_order_release);
            handle_release(hs);
            hs->batch_setting_up.store(false, std::memory_order_release);
        }
    } txn{hs, true};

    /* --- Private, fallible phase --- */
    auto bs_sp = std::make_shared<BatchState>();
    bs_sp->capacity = nr;
    bs_sp->hs = hs;
    bs_sp->hs_sp = std::move(hs_sp);  /* keep handle alive for batch lifetime */
    bs_sp->entries.resize(nr);
    bs_sp->cmd_map.resize(hs->batch_queue_depth);
    /* release_scratch is sized from capacity (the immutable SetUp
     * bound), NOT n_entries (the current round's consumed count). */
    bs_sp->release_scratch.resize(nr);

    const size_t page_size = hs->ctrl->page_size;
    PRPPool& pool = bs_sp->prp_pool;
    size_t pool_bytes = UGDS_PRP_POOL_PAGES * page_size;

    uGDSError_t setup_err = UGDS_OK;
    if (posix_memalign(&pool.buf, 4096, pool_bytes) != 0) {
        setup_err = make_error(UGDS_OUT_OF_MEMORY);
        goto setup_rollback;
    }
    std::memset(pool.buf, 0, pool_bytes);

    {
        int rc = nvm_dma_map_host(&pool.dma, hs->ctrl, pool.buf, pool_bytes);
        if (!nvm_ok(rc)) {
            free(pool.buf);
            pool.buf = nullptr;
            setup_err = (rc == ENOMEM)
                ? make_error(UGDS_OUT_OF_MEMORY)
                : make_error(UGDS_INTERNAL_ERROR);
            goto setup_rollback;
        }
    }
    pool.n_pages = UGDS_PRP_POOL_PAGES;
    pool.free_bitmap = (UGDS_PRP_POOL_PAGES >= 64)
        ? ~0ULL : (1ULL << UGDS_PRP_POOL_PAGES) - 1;

    /* --- Publication: one g_driver.lock critical section --- */
    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        if (hs->closing.load(std::memory_order_acquire)) {
            /* Deregister/force arrived mid-setup; roll back.
             * batch_setting_up is cleared as the LAST rollback action. */
            setup_err = make_error(UGDS_INVALID_VALUE);
            goto setup_rollback;
        }
        /* active_batch must be null: the CAS above ensures no other setup
         * claimed batch_active, and no destroy can have run yet. */
        if (hs->active_batch != nullptr) {
            setup_err = make_error(UGDS_INTERNAL_ERROR);
            goto setup_rollback;
        }
        bs_sp->self = bs_sp;            /* public-handle ownership */
        hs->active_batch = bs_sp;       /* owning recovery link */
        hs->batch_setting_up.store(false, std::memory_order_release);
        txn.armed = false;           /* commit: gate items disarmed */
    }

    *batch = static_cast<uGDSBatchHandle_t>(bs_sp.get());
    return UGDS_OK;

setup_rollback:
    /* Rollback in reverse acquisition order:
     * 5 (pool DMA) -> 4 (pool buf) -> 3 (BatchState shared_ptr)
     * -> 2 (batch_active) -> 1 (handle ref + batch_setting_up LAST).
     * pool.dma and pool.buf are cleaned up above or here; the shared_ptr
     * reset handles item 3. */
    {
        if (pool.dma) { nvm_dma_unmap(pool.dma); pool.dma = nullptr; }
        if (pool.buf) { free(pool.buf); pool.buf = nullptr; }
    }
    bs_sp.reset();  /* RAII: releases BatchState if last owner */
    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        hs->batch_active.store(false, std::memory_order_release);
        handle_release(hs);
        /* batch_setting_up cleared LAST so a waiting force teardown
         * proceeds only after every other side effect (including the
         * DMA unmap, which needs hs->ctrl alive) has been undone. */
        hs->batch_setting_up.store(false, std::memory_order_release);
    }
    txn.armed = false;  /* explicit rollback done; prevent double-run */
    return setup_err;
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}

extern "C" uGDSError_t uGDSBatchIOSubmit(uGDSBatchHandle_t batch, unsigned nr,
                                           uGDSIOParams_t* iocb, unsigned flags)
{
    try {
    if (batch == nullptr || iocb == nullptr || nr == 0)
        return make_error(UGDS_INVALID_VALUE);

    BatchState* bs = static_cast<BatchState*>(batch);
    std::lock_guard<std::mutex> batch_lock(bs->lock);

    if (bs->hs->wedged.load(std::memory_order_acquire) ||
        bs->hs->closing.load(std::memory_order_acquire))
        return make_error(UGDS_BUSY);

    if (bs->n_entries > 0 && bs->n_events_read == bs->n_entries) {
        bs->n_entries = 0;
        bs->n_completed = 0;
        bs->n_events_read = 0;
    }

    if (bs->n_entries + nr > bs->capacity)
        return make_error(UGDS_BATCH_CAPACITY_EXCEEDED);

    HandleState* hs = bs->hs;
    IOQueuePair& qp = hs->batch_qp->qp;
    const size_t page_size = hs->ctrl->page_size;
    const size_t max_xfer = compute_max_xfer(hs);

    (void)flags;

    // Validate and populate entries
    unsigned base = bs->n_entries;
    for (unsigned i = 0; i < nr; ++i) {
        const uGDSIOParams_t& p = iocb[i];
        BatchIOEntry& entry = bs->entries[base + i];

        if (p.devPtr_base == nullptr || p.size == 0)
            return make_error(UGDS_INVALID_VALUE);
        if (p.file_offset < 0 || p.devPtr_offset < 0)
            return make_error(UGDS_INVALID_VALUE);
        if ((static_cast<size_t>(p.file_offset) % hs->block_size) != 0 ||
            (p.size % hs->block_size) != 0)
            return make_error(UGDS_INVALID_VALUE);

        uint8_t opcode;
        if (p.opcode == UGDS_READ) opcode = NVM_IO_READ;
        else if (p.opcode == UGDS_WRITE) opcode = NVM_IO_WRITE;
        else return make_error(UGDS_INVALID_VALUE);

        entry.cookie = p.cookie;
        entry.devPtr_base = p.devPtr_base;
        entry.file_offset = p.file_offset;
        entry.devPtr_offset = p.devPtr_offset;
        entry.size = p.size;
        entry.opcode = opcode;
        entry.status = UGDS_BATCH_WAITING;
        entry.bytes_done = 0;
        entry.error_code = 0;
        entry.n_cmds_done = 0;
        entry.event_returned = false;
        /* SGL/lifecycle fields */
        entry.kind = BATCH_ENTRY_PLAIN;
        entry.seg_begin = 0;
        entry.seg_count = 0;
        entry.refs_held = false;
        entry.release_queued = false;
        entry.n_cmds_submitted = 0;

        /* Overflow-safe ceil division for n_cmds */
        size_t n_cmds = (p.size - 1) / max_xfer + 1;
        if (n_cmds > UINT16_MAX) {
            return make_error(UGDS_INVALID_VALUE);
        }
        entry.n_cmds = static_cast<uint16_t>(n_cmds);
    }

    // Build sub-command list
    struct SubCmd {
        unsigned io_idx;
        uint64_t lba;
        size_t   page_start;
        size_t   chunk_size;
        nvm_dma_t* buf_dma;
    };

    size_t total_cmds = 0;
    for (unsigned i = 0; i < nr; ++i)
        total_cmds += bs->entries[base + i].n_cmds;
    std::vector<SubCmd> work;
    work.reserve(total_cmds);

    for (unsigned i = 0; i < nr; ++i) {
        unsigned idx = base + i;
        BatchIOEntry& entry = bs->entries[idx];

        nvm_dma_t* buf_dma = nullptr;
        size_t buf_page_start = 0;
        {
            std::lock_guard<std::mutex> drv_lock(g_driver.lock);
            auto it = g_driver.buf_registry.find(entry.devPtr_base);
            if (it != g_driver.buf_registry.end()) {
                /* Controller affinity check. */
                if (it->second.map_ctrl != hs->ctrl) {
                    buf_dma = nullptr;
                }
                /* Exact-length bounds. */
                else {
                    const uint64_t off =
                        static_cast<uint64_t>(entry.devPtr_offset);
                    const uint64_t sz  =
                        static_cast<uint64_t>(entry.size);
                    if (off > it->second.length ||
                        sz > it->second.length - off) {
                        buf_dma = nullptr;
                    } else {
                        buf_dma = it->second.dma;
                    }
                }
                /* Hold in-flight reference until batch completions are drained
                 * or destroy finishes. Prevents Deregister from unmapping
                 * while batch commands retain PRPs from this buffer. */
                it->second.in_flight.fetch_add(1, std::memory_order_acq_rel);
                /* Mark that this entry holds a deferred in-flight ref. */
                entry.refs_held = true;
            }
        }

        if (buf_dma == nullptr) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
            /* Use deferred release so refs_held is consistent. */
            queue_entry_release(bs, idx);
            continue;
        }

        if ((static_cast<size_t>(entry.devPtr_offset) % page_size) != 0) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
            queue_entry_release(bs, idx);
            continue;
        }
        buf_page_start = static_cast<size_t>(entry.devPtr_offset) / page_size;

        /* Defensive page-count bounds check (subsumed by exact-length check). */
        size_t batch_pages_needed = (entry.size - 1) / page_size + 1;
        if (batch_pages_needed > buf_dma->n_ioaddrs ||
            buf_page_start > buf_dma->n_ioaddrs - batch_pages_needed) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
            queue_entry_release(bs, idx);
            continue;
        }

        uint64_t current_lba = static_cast<uint64_t>(entry.file_offset) / hs->block_size;
        size_t current_page = buf_page_start;
        size_t remaining = entry.size;

        while (remaining > 0) {
            size_t chunk = std::min(remaining, max_xfer);
            chunk = (chunk / hs->block_size) * hs->block_size;
            size_t n_pages_chunk = (chunk + page_size - 1) / page_size;
            size_t n_blocks = chunk / hs->block_size;

            work.push_back({idx, current_lba, current_page, chunk, buf_dma});

            current_lba += n_blocks;
            current_page += n_pages_chunk;
            remaining -= chunk;
        }
    }

    // Enqueue all NVMe commands to the single batch QP
    {
        std::lock_guard<std::mutex> qp_lock(qp.lock);

        for (auto& sc : work) {
            BatchIOEntry& entry = bs->entries[sc.io_idx];
            size_t n_pages = (sc.chunk_size + page_size - 1) / page_size;
            if (n_pages == 0) n_pages = 1;

            uint16_t prp_idx = UINT16_MAX;
            if (n_pages > 2) {
                int pidx = prp_pool_alloc(&bs->prp_pool);
                if (pidx < 0) {
                    nvm_sq_submit(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    uint64_t spins = 0;
                    const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
                    while ((pidx = prp_pool_alloc(&bs->prp_pool)) < 0) {
                        if (!drain_one_completion(qp, bs)) {
                            if (++spins > max_spins) {
                                /* Abort current and later entries via
                                 * the common helper. WAITING + PENDING
                                 * are both handled. Ring the SQ doorbell
                                 * for the accepted prefix. */
                                nvm_sq_submit(&qp.sq);
                                std::atomic_thread_fence(std::memory_order_seq_cst);
                                for (unsigned i = sc.io_idx; i < base + nr; ++i) {
                                    abort_phase3_entry(bs, i);
                                }
                                bs->n_entries += nr;
                                goto phase3_abort_return;
                            }
                            __builtin_ia32_pause();
                        }
                    }
                }
                prp_idx = static_cast<uint16_t>(pidx);
            }

            uint16_t slot = static_cast<uint16_t>(
                qp.sq.tail.load(std::memory_order_relaxed) % qp.sq.qs);
            nvm_cmd_t* cmd = nvm_sq_enqueue(&qp.sq);

            if (cmd == nullptr) {
                nvm_sq_submit(&qp.sq);
                std::atomic_thread_fence(std::memory_order_seq_cst);

                uint64_t spins = 0;
                const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
                bool drained = false;
                while (!drained) {
                    if (drain_one_completion(qp, bs)) {
                        drained = true;
                    } else if (++spins > max_spins) {
                        if (prp_idx != UINT16_MAX)
                            prp_pool_free(&bs->prp_pool, prp_idx);
                        /* Abort current and later entries. */
                        nvm_sq_submit(&qp.sq);
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        for (unsigned i = sc.io_idx; i < base + nr; ++i) {
                            abort_phase3_entry(bs, i);
                        }
                        bs->n_entries += nr;
                        goto phase3_abort_return;
                    } else {
                        __builtin_ia32_pause();
                    }
                }

                slot = static_cast<uint16_t>(
                    qp.sq.tail.load(std::memory_order_relaxed) % qp.sq.qs);
                cmd = nvm_sq_enqueue(&qp.sq);
                if (cmd == nullptr) {
                    if (prp_idx != UINT16_MAX)
                        prp_pool_free(&bs->prp_pool, prp_idx);
                    /* Abort current and later entries. */
                    nvm_sq_submit(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    for (unsigned i = sc.io_idx; i < base + nr; ++i) {
                        abort_phase3_entry(bs, i);
                    }
                    bs->n_entries += nr;
                    goto phase3_abort_return;
                }
            }

            memset(cmd, 0, sizeof(nvm_cmd_t));
            nvm_cmd_header(cmd, slot, entry.opcode, hs->ns_id);

            if (n_pages == 1) {
                nvm_cmd_data_ptr(cmd, sc.buf_dma->ioaddrs[sc.page_start], 0);
            } else if (n_pages == 2) {
                nvm_cmd_data_ptr(cmd,
                    sc.buf_dma->ioaddrs[sc.page_start],
                    sc.buf_dma->ioaddrs[sc.page_start + 1]);
            } else {
                volatile uint64_t* prp_list = reinterpret_cast<volatile uint64_t*>(
                    static_cast<uint8_t*>(bs->prp_pool.buf) + prp_idx * page_size);
                for (size_t p = 1; p < n_pages; ++p) {
                    prp_list[p - 1] = sc.buf_dma->ioaddrs[sc.page_start + p];
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                nvm_cmd_data_ptr(cmd,
                    sc.buf_dma->ioaddrs[sc.page_start],
                    bs->prp_pool.dma->ioaddrs[prp_idx]);
            }

            size_t n_blocks = sc.chunk_size / hs->block_size;
            nvm_cmd_rw_blks(cmd, sc.lba, static_cast<uint16_t>(n_blocks));

            CmdSlot& cs = bs->cmd_map[slot];
            cs.io_idx = static_cast<uint16_t>(sc.io_idx);
            cs.chunk_bytes = sc.chunk_size;
            cs.prp_page_idx = prp_idx;
            cs.active = true;
            bs->in_flight++;

            /* M2: commit the slot map first, then increment
             * n_cmds_submitted, then transition WAITING->PENDING. */
            entry.n_cmds_submitted++;
            entry.status = UGDS_BATCH_PENDING;
        }

        // Single doorbell for all commands
        std::atomic_thread_fence(std::memory_order_seq_cst);
        nvm_sq_submit(&qp.sq);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    /* Drain deferred releases after qp.lock is released. */
    drain_release_scratch(bs);

    bs->n_entries += nr;
    return UGDS_OK;

phase3_abort_return:
    /* Drain deferred releases after qp.lock is released.  The abort
     * helper queued WAITING/PENDING entries for release. */
    drain_release_scratch(bs);
    return make_error(UGDS_INTERNAL_ERROR);
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}

extern "C" uGDSError_t uGDSBatchIOGetStatus(uGDSBatchHandle_t batch,
                                              unsigned min_nr, unsigned* nr,
                                              uGDSIOEvents_t* events,
                                              struct timespec* timeout)
{
    if (batch == nullptr || nr == nullptr || events == nullptr)
        return make_error(UGDS_INVALID_VALUE);

    BatchState* bs = static_cast<BatchState*>(batch);
    HandleState* hs = bs->hs;
    IOQueuePair& qp = hs->batch_qp->qp;
    unsigned max_events = *nr > 0 ? *nr : bs->capacity;

    bool has_deadline = false;
    struct timespec deadline;
    if (timeout != nullptr) {
        has_deadline = true;
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += timeout->tv_sec;
        deadline.tv_nsec += timeout->tv_nsec;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000L;
        }
    }

    unsigned n_ready = 0;

    while (true) {
        std::lock_guard<std::mutex> batch_lock(bs->lock);

        // Poll single CQ for completions
        if (bs->in_flight > 0) {
            std::lock_guard<std::mutex> qp_lock(qp.lock);
            while (drain_one_completion(qp, bs)) {}
        }

        /* Drain deferred releases after qp.lock is released. */
        drain_release_scratch(bs);

        for (unsigned i = 0; i < bs->n_entries && n_ready < max_events; ++i) {
            BatchIOEntry& entry = bs->entries[i];
            if (entry.event_returned) continue;
            if (entry.status != UGDS_BATCH_COMPLETE &&
                entry.status != UGDS_BATCH_FAILED) continue;

            events[n_ready].cookie = entry.cookie;
            events[n_ready].status = entry.status;
            events[n_ready].ret = entry.error_code;
            entry.event_returned = true;
            bs->n_events_read++;
            n_ready++;
        }

        if (n_ready >= min_nr || min_nr == 0 || n_ready >= max_events) {
            *nr = n_ready;
            return UGDS_OK;
        }

        if (has_deadline) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec > deadline.tv_sec ||
                (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
                *nr = n_ready;
                return UGDS_OK;
            }
        }

        __builtin_ia32_pause();
    }
}

extern "C" void uGDSBatchIODestroy(uGDSBatchHandle_t batch)
{
    if (batch == nullptr) return;

    BatchState* bs = static_cast<BatchState*>(batch);
    HandleState* hs = bs->hs;

    /* Pin rule: declare pins BEFORE the lock guard so they outlive
     * it. C++ destroys locals in reverse construction order: the
     * guard unlocks before the pins drop. This prevents ~BatchState
     * from running while the mutex is still locked. */
    std::shared_ptr<BatchState> last_pin;
    std::shared_ptr<BatchState> link_pin;

    {
        std::lock_guard<std::mutex> bl(bs->lock);

        /* Tombstone check: if a force teardown already set TORN_DOWN,
         * this is the post-force one-shot Destroy. Transfer self into
         * a local pin and let the guard unlock before the pin drops. */
        if (bs->lifecycle == BATCH_LIFECYCLE_TORN_DOWN) {
            last_pin = std::move(bs->self);
            return;
        }

        /* Drain remaining in-flight commands under bs->lock -> qp.lock. */
        IOQueuePair& qp = hs->batch_qp->qp;
        if (bs->in_flight > 0) {
            std::lock_guard<std::mutex> qp_lock(qp.lock);
            uint64_t spins = 0;
            const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
            while (bs->in_flight > 0) {
                if (!drain_one_completion(qp, bs)) {
                    if (++spins > max_spins) break;
                    __builtin_ia32_pause();
                } else {
                    spins = 0;
                }
            }
        }

        if (bs->in_flight > 0) {
            /* Wedge: commands still in flight after drain timeout.
             * Retain everything (self, link, handle ref, batch_active)
             * so force recovery can find the object. Transition to
             * WEDGED and return -- the one-shot Destroy was consumed. */
            bs->lifecycle = BATCH_LIFECYCLE_WEDGED;
            hs->wedged.store(true, std::memory_order_release);
            fprintf(stderr, "uGDS: BatchIODestroy: %u commands still in "
                    "flight after drain timeout -- handle is now wedged "
                    "to prevent DMA-after-unmap. Controller reset required.\n",
                    bs->in_flight);
            return;
        }

        /* --- Resource release (allocation-free) --- */

        /* Drain any live release_scratch prefix first. */
        drain_release_scratch(bs);

        /* Scan for remaining refs_held entries and queue them. */
        for (unsigned i = 0; i < bs->n_entries; ++i) {
            BatchIOEntry& entry = bs->entries[i];
            if (entry.refs_held && !entry.release_queued) {
                queue_entry_release(bs, i);
            }
        }
        /* Final drain for remaining held entries. */
        drain_release_scratch(bs);

        cleanup_prp_pool(bs);
        bs->lifecycle = BATCH_LIFECYCLE_TORN_DOWN;

        /* --- Link transaction: one g_driver.lock section --- */
        {
            std::lock_guard<std::mutex> g(g_driver.lock);
            if (hs->active_batch.get() == bs) {
                /* Pointer-verified detach: move OUR link into a local pin,
                 * then clear batch_active ONLY after the detach. */
                link_pin = std::move(hs->active_batch);
                hs->batch_active.store(false, std::memory_order_release);
            }
            /* else: a concurrent force walk already moved the link.
             * Do NOT clear batch_active -- the handle is closing. */
        }

        /* Release the SetUp-time handle reference. hs stays alive via
         * bs->hs_sp. */
        handle_release(hs);

        /* Pin rule: transfer self into last_pin. The object stays alive
         * until the guard releases the mutex and last_pin drops. */
        last_pin = std::move(bs->self);
    }  /* bl unlocks here -- object alive via last_pin (+ link_pin) */

    /* link_pin then last_pin drop here; whichever is the final reference
     * runs ~BatchState with no guard addressing its members. */
}
