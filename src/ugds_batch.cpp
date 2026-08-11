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

#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(((size_t)1 << (sizeof(ssize_t) * 8 - 1)) - 1))
#endif

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

    /* BatchSetupTxn - RAII scope guard that owns ALL five items by value.
     * Pool resources are owned by value (not pointer to bs_sp members)
     * so destruction order does not cause dangling dereferences.
     * On commit, pool resources are transferred to bs_sp->prp_pool
     * and txn.armed is cleared.
     * Destructor uses a blocking lock_guard because gate-state transitions
     * must be atomic under g_driver.lock. If the lock throws (system_error),
     * std::terminate is the correct outcome: the process state is
     * unrecoverable. try_to_lock would silently skip the mutex and race. */
    struct BatchSetupTxn {
        HandleState* hs;
        void* pool_buf;
        nvm_dma_t* pool_dma;
        bool armed;
        ~BatchSetupTxn() {
            if (!armed) return;
            /* Rollback items 5, 4 (owned by value) */
            if (pool_dma) { nvm_dma_unmap(pool_dma); }
            if (pool_buf) { free(pool_buf); }
            /* Rollback items 3, 2, 1 (gate state under g_driver.lock) */
            std::lock_guard<std::mutex> g(g_driver.lock);
            hs->batch_active.store(false, std::memory_order_release);
            handle_release(hs);
            hs->batch_setting_up.store(false, std::memory_order_release);
        }
    } txn{hs, nullptr, nullptr, true};

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
    size_t pool_bytes = UGDS_PRP_POOL_PAGES * page_size;

    uGDSError_t setup_err = UGDS_OK;
    void* pbuf = nullptr;
    nvm_dma_t* pdma = nullptr;
    if (posix_memalign(&pbuf, 4096, pool_bytes) != 0) {
        setup_err = make_error(UGDS_OUT_OF_MEMORY);
        goto setup_rollback;
    }
    std::memset(pbuf, 0, pool_bytes);
    txn.pool_buf = pbuf;  /* txn owns the buffer now */

    {
        int rc = nvm_dma_map_host(&pdma, hs->ctrl, pbuf, pool_bytes);
        if (!nvm_ok(rc)) {
            txn.pool_buf = nullptr;  /* txn no longer owns; we free below */
            free(pbuf);
            pbuf = nullptr;
            setup_err = (rc == ENOMEM)
                ? make_error(UGDS_OUT_OF_MEMORY)
                : make_error(UGDS_INTERNAL_ERROR);
            goto setup_rollback;
        }
    }
    txn.pool_dma = pdma;  /* txn owns the mapping now */
    bs_sp->prp_pool.n_pages = UGDS_PRP_POOL_PAGES;
    bs_sp->prp_pool.free_bitmap = (UGDS_PRP_POOL_PAGES >= 64)
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
        /* Transfer pool resources from txn to bs_sp */
        bs_sp->prp_pool.buf = txn.pool_buf;
        bs_sp->prp_pool.dma = txn.pool_dma;
        txn.armed = false;       /* commit: txn no longer owns anything */
        bs_sp->self = bs_sp;            /* public-handle ownership */
        hs->active_batch = bs_sp;       /* owning recovery link */
        hs->batch_setting_up.store(false, std::memory_order_release);
    }

    *batch = static_cast<uGDSBatchHandle_t>(bs_sp.get());
    return UGDS_OK;

setup_rollback:
    /* Clean up pool resources first, then gate state. txn stays armed
     * so its destructor handles gate rollback if the lock_guard below
     * throws (the destructor will block on the same lock, which is fine
     * since the throwing lock_guard's exception unwinds this frame). */
    {
        if (txn.pool_dma) { nvm_dma_unmap(txn.pool_dma); txn.pool_dma = nullptr; }
        if (txn.pool_buf) { free(txn.pool_buf); txn.pool_buf = nullptr; }
    }
    bs_sp.reset();
    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        hs->batch_active.store(false, std::memory_order_release);
        handle_release(hs);
        hs->batch_setting_up.store(false, std::memory_order_release);
    }
    txn.armed = false;  /* gate cleanup done; disarm to prevent double-run */
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
        /* Reset the segment arena allocation point.
         * seg_arena.size()/capacity are never touched here; only the
         * allocation cursor resets so the next Submitv rewrites from
         * index 0.  Plain-only batches never resized the arena and
         * this store is a harmless no-op on the zero-initialized
         * value. */
        bs->arena_used = 0;
    }

    /* Overflow-safe capacity check.  The old n_entries + nr >
     * capacity used unsigned addition that wraps for large nr.
     * Reject nr > capacity outright, then check the post-recycle
     * base.  At this point n_entries is already recycled if the
     * round was fully consumed, so n_entries is the effective base. */
    if (nr > bs->capacity ||
        bs->n_entries > bs->capacity - nr)
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
    try {
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
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}

extern "C" void uGDSBatchIODestroy(uGDSBatchHandle_t batch)
{
    if (batch == nullptr) return;
    try {
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
    } catch (...) {
        /* Destroy is void; swallow to prevent crossing C ABI.
         * The BatchState may be partially cleaned up but shared_ptr
         * ref counts prevent double-free. */
    }
}

/* ======================================================================== */
/* uGDSBatchIOSubmitv: vectored batch submit                              */
/* ======================================================================== */

/* Registry preflight result: a stack record filled under g_driver.lock
 * without taking refs.  Classifies each entry so the caller can size
 * the work array before any ref acquisition or allocation. */
struct PreflightResult {
    bool        valid;
};

/* Pure value validation for one vectored entry.
 * Performs the value-matrix checks and computes the analytic window
 * count via sgl_count_windows_analytic.  Writes nothing to bs.
 * Returns true on success (out_n_cmds set), false on value error. */
static bool submitv_validate_entry(const uGDSIOSegParams_t& p,
                                   size_t page_size, size_t block_size,
                                   size_t window_cap, size_t SU,
                                   uint32_t* out_n_cmds)
{
    if (p.segs == nullptr || p.nr_segs == 0)
        return false;
    if (p.nr_segs > UGDS_BATCH_IOV_MAX)
        return false;
    if (p.file_offset < 0)
        return false;
    if ((static_cast<size_t>(p.file_offset) % block_size) != 0)
        return false;

    /* opcode check */
    if (p.opcode != UGDS_READ && p.opcode != UGDS_WRITE)
        return false;

    uint64_t total_size = 0;
    for (unsigned k = 0; k < p.nr_segs; ++k) {
        if (p.segs[k].base == nullptr || p.segs[k].size == 0)
            return false;
        if (p.segs[k].offset < 0)
            return false;
        if ((static_cast<size_t>(p.segs[k].offset) % page_size) != 0)
            return false;
        if ((p.segs[k].size % block_size) != 0)
            return false;

        uint64_t seg_size = static_cast<uint64_t>(p.segs[k].size);
        if (seg_size > static_cast<uint64_t>(SSIZE_MAX) - total_size)
            return false;  /* total overflow */
        total_size += seg_size;
    }

    /* Analytic window count over the segment array.  O(nr_segs). */
    uint32_t n_cmds = 0;
    if (!sgl_count_windows_analytic(p.segs, p.nr_segs,
                                    window_cap, page_size, block_size,
                                    &n_cmds))
        return false;

    if (n_cmds == 0)
        return false;

    /* BatchIOEntry::n_cmds is uint16_t.  Reject counts that would
     * truncate before any allocation or commit-copy. */
    if (n_cmds > UINT16_MAX)
        return false;

    (void)SU;  /* window_cap already incorporates SU rounding */
    *out_n_cmds = n_cmds;
    return true;
}

extern "C" uGDSError_t uGDSBatchIOSubmitv(uGDSBatchHandle_t batch, unsigned nr,
                                            uGDSIOSegParams_t* iocb,
                                            unsigned flags)
{
    try {
    /* ==================================================================
     * Basic call/batch checks (no state mutation)
     * ================================================================== */
    if (batch == nullptr || iocb == nullptr || nr == 0)
        return make_error(UGDS_INVALID_VALUE);

    BatchState* bs = static_cast<BatchState*>(batch);
    std::lock_guard<std::mutex> batch_lock(bs->lock);

    if (bs->hs->wedged.load(std::memory_order_acquire) ||
        bs->hs->closing.load(std::memory_order_acquire))
        return make_error(UGDS_BUSY);

    if (bs->lifecycle != BATCH_LIFECYCLE_ACTIVE)
        return make_error(UGDS_INVALID_VALUE);

    /* Overflow-safe capacity check with an effective base that accounts
     * for a fully consumed recyclable round, WITHOUT mutating any
     * state.  The old code computed n_entries + nr using unsigned
     * addition (wraps for large nr) and ran recycle before this check,
     * so a full-capacity batch rejected any new submit and a later
     * value error violated consume-none.  We compute the effective base
     * here and defer the actual recycle reset to the commit step after
     * all validation, preflight, and allocation succeed. */
    unsigned effective_base = bs->n_entries;
    if (effective_base > 0 && bs->n_events_read == effective_base)
        effective_base = 0;  /* round is fully consumed; would recycle */

    if (nr > bs->capacity ||
        effective_base > bs->capacity - nr)
        return make_error(UGDS_BATCH_CAPACITY_EXCEEDED);

    (void)flags;

    HandleState* hs = bs->hs;
    IOQueuePair& qp = hs->batch_qp->qp;
    const size_t page_size  = hs->ctrl->page_size;
    const size_t block_size = hs->block_size;
    const size_t max_xfer   = compute_max_xfer(hs);

    /* Compute SU and window_cap (parity with sync vectored path). */
    const size_t SU = (page_size > block_size &&
                       (page_size & (page_size - 1)) == 0 &&
                       (block_size & (block_size - 1)) == 0)
                      ? std::max(page_size, block_size)
                      : [&]() {
                          size_t a = page_size, b = block_size;
                          while (b != 0) { size_t t = a % b; a = b; b = t; }
                          return (page_size / a) * block_size;
                      }();
    size_t window_cap = (max_xfer / SU) * SU;
    if (window_cap == 0)
        return make_error(UGDS_INVALID_VALUE);

    /* ==================================================================
     * Validate -> preflight -> allocate -> commit-copy
     * ================================================================== */

    /* --- Step 1: Pure value validation (all entries, no state write) --- */
    /* Use effective_base, which accounts for a recyclable round
     * without mutating bs state.  The actual recycle reset is applied
     * as part of the infallible commit (Step 4) after all validation,
     * preflight, and allocation succeed. */
    unsigned base = effective_base;

    uint32_t per_entry_n_cmds[UGDS_MAX_BATCH_IO_SIZE];
    uint64_t total_cmds = 0;
    for (unsigned i = 0; i < nr; ++i) {
        uint32_t n_cmds_i = 0;
        if (!submitv_validate_entry(iocb[i], page_size, block_size,
                                    window_cap, SU, &n_cmds_i))
            return make_error(UGDS_INVALID_VALUE);

        /* checked addition for total_cmds */
        if (total_cmds > UINT64_MAX - n_cmds_i)
            return make_error(UGDS_INVALID_VALUE);
        total_cmds += n_cmds_i;
        per_entry_n_cmds[i] = n_cmds_i;
    }

    if (total_cmds > UINT32_MAX)
        return make_error(UGDS_INVALID_VALUE);

    /* --- Step 2: Registry preflight (read-only, no refs) ---
     * Classifies each entry; failures become terminal FAILED -EINVAL
     * and contribute zero to alloc_cmds. */
    PreflightResult preflight[UGDS_MAX_BATCH_IO_SIZE];
    uint64_t alloc_cmds = 0;
    {
        std::lock_guard<std::mutex> drv_lock(g_driver.lock);
        for (unsigned i = 0; i < nr; ++i) {
            const uGDSIOSegParams_t& p = iocb[i];
            bool ok = true;

            /* Check every segment: registration, exact-length, affinity. */
            for (unsigned k = 0; k < p.nr_segs && ok; ++k) {
                auto it = g_driver.buf_registry.find(p.segs[k].base);
                if (it == g_driver.buf_registry.end()) {
                    ok = false;
                    break;
                }
                DriverState::BufEntry& be = it->second;
                /* Controller affinity check */
                if (be.map_ctrl != hs->ctrl) { ok = false; break; }
                /* Exact-length bounds */
                const uint64_t off = static_cast<uint64_t>(p.segs[k].offset);
                const uint64_t sz  = static_cast<uint64_t>(p.segs[k].size);
                if (off > be.length || sz > be.length - off) {
                    ok = false;
                    break;
                }
            }

            preflight[i].valid = ok;
            if (ok) {
                alloc_cmds += per_entry_n_cmds[i];
            }
        }
    }

    /* --- Step 3: Allocate (fallible, still no entry/watermark mutation) ---
     * First the local work array, then the one-time arena resize.
     * The arena resize is the final fallible operation before commit. */
    std::vector<SubCmdV> work;
    if (alloc_cmds > 0)
        work.resize(static_cast<size_t>(alloc_cmds));

    if (bs->seg_arena.empty()) {
        /* One lifetime allocation: capacity * UGDS_BATCH_IOV_MAX.
         * capacity <= 128, UGDS_BATCH_IOV_MAX = 128, so <= 16384
         * SegView elements (~48 B each = ~768 KiB). */
        size_t arena_elems = static_cast<size_t>(bs->capacity) *
                             static_cast<size_t>(UGDS_BATCH_IOV_MAX);
        bs->seg_arena.resize(arena_elems);
    }

    /* --- Step 4: Commit-copy (infallible) ---
     * From here, no allocation or throw is possible.  Capture the
     * watermark for rollback safety; populate entries and arena.
     * The watermark is the rollback anchor: it is never restored in
     * this path because no operation after this point can fail, but it
     * documents the invariant and serves as the audit point for the
     * fixed-size arena model.
     *
     * Apply the recycle reset here as the first infallible commit
     * action.  effective_base was 0 iff the current round was fully
     * consumed.  This is the consume-none guarantee: a value error
     * returns without touching n_entries, n_completed, n_events_read,
     * or arena_used. */
    if (effective_base == 0 && bs->n_entries > 0) {
        bs->n_entries = 0;
        bs->n_completed = 0;
        bs->n_events_read = 0;
        bs->arena_used = 0;
    }
    const uint32_t watermark = bs->arena_used;
    (void)watermark;  /* documented rollback anchor; no post-commit failure */
    uint32_t work_used = 0;

    for (unsigned i = 0; i < nr; ++i) {
        unsigned idx = base + i;
        BatchIOEntry& entry = bs->entries[idx];
        const uGDSIOSegParams_t& p = iocb[i];

        /* Clear entry fields for both success and failure paths. */
        entry.cookie = p.cookie;
        entry.devPtr_base = nullptr;   /* not used for vectored */
        entry.file_offset = p.file_offset;
        entry.devPtr_offset = 0;
        entry.size = 0;
        entry.opcode = (p.opcode == UGDS_READ) ? NVM_IO_READ : NVM_IO_WRITE;
        entry.status = UGDS_BATCH_WAITING;
        entry.bytes_done = 0;
        entry.error_code = 0;
        entry.n_cmds_done = 0;
        entry.event_returned = false;
        entry.refs_held = false;
        entry.release_queued = false;
        entry.n_cmds_submitted = 0;

        if (!preflight[i].valid) {
            /* Preflight failure: terminal FAILED -EINVAL, n_cmds = 0. */
            entry.kind = BATCH_ENTRY_VECTORED;
            entry.seg_begin = 0;
            entry.seg_count = 0;
            entry.n_cmds = 0;
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            bs->n_completed++;
            continue;
        }

        /* Preflight-valid entry: copy segments into the arena. */
        assert(bs->arena_used + p.nr_segs <= bs->seg_arena.size());
        entry.kind = BATCH_ENTRY_VECTORED;
        entry.seg_begin = bs->arena_used;
        entry.seg_count = static_cast<uint32_t>(p.nr_segs);
        entry.n_cmds = static_cast<uint16_t>(per_entry_n_cmds[i]);

        for (unsigned k = 0; k < p.nr_segs; ++k) {
            bs->seg_arena[bs->arena_used + k] = SegView{};
        }
        bs->arena_used += static_cast<uint32_t>(p.nr_segs);
    }

    /* ==================================================================
     * Acquiring resolve / build
     * ==================================================================
     * For each preflight-valid entry, acquire refs (all-or-nothing per
     * entry via SglRefOwner), resolve SegView geometry into the arena,
     * and stream windows into the work array via SglWindowCursor. */

    for (unsigned i = 0; i < nr; ++i) {
        unsigned idx = base + i;
        BatchIOEntry& entry = bs->entries[idx];

        /* Skip entries already terminalized in commit-copy. */
        if (entry.status == UGDS_BATCH_FAILED)
            continue;

        const uGDSIOSegParams_t& p = iocb[i];
        SegView* seg_views = bs->seg_arena.data() + entry.seg_begin;

        /* Acquire in-flight refs and resolve geometry (all-or-nothing). */
        SglRefOwner owner;
        int rc = owner.acquire(p.segs, static_cast<uint32_t>(p.nr_segs),
                               hs->ctrl, page_size, seg_views);
        if (rc != 0) {
            /* Resolve failure: entry-local.  No work items emitted. */
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            entry.n_cmds_submitted = 0;
            bs->n_completed++;
            /* owner is empty (all-or-nothing), nothing to release. */
            continue;
        }

        /* refs are now held by the arena; mark for deferred release. */
        entry.refs_held = true;

        /* Stream windows into the work array. */
        SglWindowCursor cursor;
        sgl_cursor_init(cursor, seg_views, static_cast<uint32_t>(p.nr_segs),
                        window_cap, page_size, block_size);

        uint64_t current_lba =
            static_cast<uint64_t>(p.file_offset) / block_size;

        CmdWindow win;
        while (sgl_cursor_next(cursor, win)) {
            assert(work_used < work.size());
            work[work_used].io_idx = idx;
            work[work_used].lba    = current_lba;
            work[work_used].window = win;
            work_used++;

            current_lba += win.bytes / block_size;
        }
    }

    /* ==================================================================
     * Enqueue (mirrors the plain submit)
     * ================================================================== */
    {
        std::lock_guard<std::mutex> qp_lock(qp.lock);

        for (uint32_t wi = 0; wi < work_used; ++wi) {
            SubCmdV& sc = work[wi];
            BatchIOEntry& entry = bs->entries[sc.io_idx];

            const size_t n_pages = sc.window.n_pages;
            SegView* seg_views = bs->seg_arena.data() + entry.seg_begin;

            /* PRP pool page for windows needing a list (> 2 pages). */
            uint16_t prp_idx = UINT16_MAX;
            if (n_pages > 2) {
                int pidx = prp_pool_alloc(&bs->prp_pool);
                if (pidx < 0) {
                    /* Pool exhaustion: submit ring, then drain loop. */
                    nvm_sq_submit(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    uint64_t spins = 0;
                    const uint64_t max_spins =
                        (uint64_t)hs->ctrl->timeout * 1000000ULL;
                    while ((pidx = prp_pool_alloc(&bs->prp_pool)) < 0) {
                        if (!drain_one_completion(qp, bs)) {
                            if (++spins > max_spins) {
                                nvm_sq_submit(&qp.sq);
                                std::atomic_thread_fence(
                                    std::memory_order_seq_cst);
                                for (unsigned i = sc.io_idx;
                                     i < base + nr; ++i)
                                    abort_phase3_entry(bs, i);
                                bs->n_entries += nr;
                                goto phase3v_abort_return;
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
                /* SQ full: submit ring, drain, retry. */
                nvm_sq_submit(&qp.sq);
                std::atomic_thread_fence(std::memory_order_seq_cst);

                uint64_t spins = 0;
                const uint64_t max_spins =
                    (uint64_t)hs->ctrl->timeout * 1000000ULL;
                bool drained = false;
                while (!drained) {
                    if (drain_one_completion(qp, bs)) {
                        drained = true;
                    } else if (++spins > max_spins) {
                        if (prp_idx != UINT16_MAX)
                            prp_pool_free(&bs->prp_pool, prp_idx);
                        nvm_sq_submit(&qp.sq);
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        for (unsigned i = sc.io_idx; i < base + nr; ++i)
                            abort_phase3_entry(bs, i);
                        bs->n_entries += nr;
                        goto phase3v_abort_return;
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
                    nvm_sq_submit(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    for (unsigned i = sc.io_idx; i < base + nr; ++i)
                        abort_phase3_entry(bs, i);
                    bs->n_entries += nr;
                    goto phase3v_abort_return;
                }
            }

            memset(cmd, 0, sizeof(nvm_cmd_t));
            nvm_cmd_header(cmd, slot, entry.opcode, hs->ns_id);

            /* Build PRP via SglPageCursor. */
            SglPageCursor pc;
            sgl_page_cursor_init(pc, seg_views,
                                 sc.window.first_seg,
                                 sc.window.first_seg_page_off,
                                 sc.window.n_segs, sc.window.n_pages);

            if (n_pages == 1) {
                uint64_t prp1 = sgl_page_cursor_next(pc);
                nvm_cmd_data_ptr(cmd, prp1, 0);
            } else if (n_pages == 2) {
                uint64_t prp1 = sgl_page_cursor_next(pc);
                uint64_t prp2 = sgl_page_cursor_next(pc);
                nvm_cmd_data_ptr(cmd, prp1, prp2);
            } else {
                volatile uint64_t* prp_list =
                    reinterpret_cast<volatile uint64_t*>(
                        static_cast<uint8_t*>(bs->prp_pool.buf) +
                        prp_idx * page_size);
                uint64_t prp1 = sgl_page_cursor_next(pc);
                for (size_t p = 1; p < n_pages; ++p)
                    prp_list[p - 1] = sgl_page_cursor_next(pc);
                sgl_publish_prp();
                nvm_cmd_data_ptr(cmd, prp1,
                                 bs->prp_pool.dma->ioaddrs[prp_idx]);
            }

            size_t n_blocks = sc.window.bytes / block_size;
            nvm_cmd_rw_blks(cmd, sc.lba, static_cast<uint16_t>(n_blocks));

            CmdSlot& cs = bs->cmd_map[slot];
            cs.io_idx = static_cast<uint16_t>(sc.io_idx);
            cs.chunk_bytes = sc.window.bytes;
            cs.prp_page_idx = prp_idx;
            cs.active = true;
            bs->in_flight++;

            /* Commit slot map, then n_cmds_submitted, then WAITING->PENDING. */
            entry.n_cmds_submitted++;
            entry.status = UGDS_BATCH_PENDING;
        }

        /* Single doorbell for all enqueued commands. */
        sgl_publish_prp();
        nvm_sq_submit(&qp.sq);
        sgl_publish_prp();
    }

    /* Drain deferred releases after qp.lock is released. */
    drain_release_scratch(bs);

    bs->n_entries += nr;
    return UGDS_OK;

phase3v_abort_return:
    /* Abort path: the abort helper queued entries for deferred release. */
    drain_release_scratch(bs);
    return make_error(UGDS_INTERNAL_ERROR);
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}
