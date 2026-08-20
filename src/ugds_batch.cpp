#include "ugds_internal.h"

#include <cstring>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <atomic>
#include <time.h>
#include <cstdio>
#include <poll.h>
#include <unistd.h>

/* Batch completion architecture
 * -----------------------------
 * Both modes decode CQEs through drain_completions().  It consumes every CQE
 * currently visible and rings the CQ doorbell once for the whole burst.
 *
 * Polling mode repeatedly drains the CQ and keeps the caller on CPU.
 * Interrupt mode drains before sleeping, blocks on the CQ's eventfd, then
 * drains again.  Checking the CQ before ppoll prevents a completion that is
 * already visible from being hidden behind a later interrupt.
 *
 * Submit may also poll while reclaiming SQ/PRP resources.  That is queue
 * backpressure inside Submit, not the completion mode selected by the user.
 */

/* Release in-flight reference held by batch submit.
 * Called on validation failure or when batch entry completes. */
static void async_release_inflight_batch(void* devPtr_base)
{
    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    auto it = g_driver.buf_registry.find(devPtr_base);
    if (it != g_driver.buf_registry.end())
        it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
}

static void prp_pool_destroy(BatchState* bs)
{
    PRPPool& pool = bs->prp_pool;
    if (pool.dma) nvm_dma_unmap(pool.dma);
    free(pool.buf);
    pool.dma = nullptr;
    pool.buf = nullptr;
}

static int prp_pool_alloc_page(PRPPool* pool)
{
    if (pool->free_bitmap == 0) return -1;
    int idx = __builtin_ctzll(pool->free_bitmap);
    pool->free_bitmap &= ~(1ULL << idx);
    return idx;
}

static void prp_pool_release_page(PRPPool* pool, int idx)
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
    //Represents which batch hardware command in the SQ
    CmdSlot& slot = bs->cmd_map[cid];
    //Represents which sub‑IO within the batch packet
    BatchIOEntry& entry = bs->entries[slot.io_idx];

    if (status != 0) {
        entry.status = UGDS_BATCH_FAILED;
    } else {
        entry.bytes_done += slot.chunk_bytes;
    }

    if (slot.prp_page_idx != UINT16_MAX) {
        prp_pool_release_page(&bs->prp_pool, slot.prp_page_idx);
    }
    slot.active = false;
    bs->in_flight--;

    entry.n_cmds_done++;

    if (entry.n_cmds_done == entry.n_cmds) {
        if (entry.status != UGDS_BATCH_FAILED) {
            entry.status = UGDS_BATCH_COMPLETE;
        }
        entry.error_code = entry.bytes_done;
        bs->n_completed++;
        /* Release in-flight reference when all sub-commands complete */
        async_release_inflight_batch(entry.devPtr_base);
    }

    return true;
}

static unsigned drain_completions(IOQueuePair& qp, BatchState* bs)
{
    unsigned drained = 0;
    while (drain_one_completion(qp, bs))
        ++drained;

    if (drained > 0) {
        /* Publish one new CQ head after the entire ready burst, rather than
         * paying one MMIO doorbell write for every CQE. */
        std::atomic_thread_fence(std::memory_order_seq_cst);
        nvm_cq_update(&qp.cq);
    }
    return drained;
}

//Calculate the maximum number of bytes that can be transferred in a single 
//uGDS hardware command
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
    if (batch == nullptr || fh == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (nr == 0 || nr > UGDS_MAX_BATCH_IO_SIZE)
        return make_error(UGDS_INVALID_VALUE);

    /* Look up handle via global registry to safely acquire a shared_ptr.
     * This prevents UAF if HandleDeregister races with us. */
    std::shared_ptr<HandleState> hs_sp;
    HandleState* hs = handle_lookup(fh, &hs_sp);
    if (!hs)
        return make_error(UGDS_INVALID_VALUE);

    if (!hs->batch_qp) {
        handle_release(hs);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    bool expected = false;
    if (!hs->batch_active.compare_exchange_strong(expected, true)) {
        handle_release(hs);
        return make_error(UGDS_INVALID_VALUE);
    }

    auto bs = new (std::nothrow) BatchState();
    if (!bs) {
        hs->batch_active.store(false);
        handle_release(hs);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    bs->capacity = nr;
    bs->hs = hs;
    bs->hs_sp = std::move(hs_sp);  /* keep handle alive for batch lifetime */
    bs->entries.resize(nr);
    bs->cmd_map.resize(hs->batch_queue_depth);

    const size_t page_size = hs->ctrl->page_size;
    PRPPool& pool = bs->prp_pool;
    size_t pool_bytes = UGDS_PRP_POOL_PAGES * page_size;

    if (posix_memalign(&pool.buf, 4096, pool_bytes) != 0) {
        delete bs;
        hs->batch_active.store(false);
        handle_release(hs);
        return make_error(UGDS_INTERNAL_ERROR);
    }
    std::memset(pool.buf, 0, pool_bytes);

    int rc = nvm_dma_map_host(&pool.dma, hs->ctrl, pool.buf, pool_bytes);
    if (!nvm_ok(rc)) {
        free(pool.buf);
        pool.buf = nullptr;
        delete bs;
        hs->batch_active.store(false);
        handle_release(hs);
        return make_error(UGDS_INTERNAL_ERROR);
    }
    pool.n_pages = UGDS_PRP_POOL_PAGES;
    pool.free_bitmap = (UGDS_PRP_POOL_PAGES >= 64)
        ? ~0ULL : (1ULL << UGDS_PRP_POOL_PAGES) - 1;

    *batch = static_cast<uGDSBatchHandle_t>(bs);

    return UGDS_OK;
}

extern "C" uGDSError_t uGDSBatchIOSubmit(uGDSBatchHandle_t batch, unsigned nr,
                                           uGDSIOParams_t* iocb, unsigned flags)
{
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

    // Phase 1: validate and populate entries
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

        /* Overflow-safe ceil division for n_cmds */
        size_t n_cmds = (p.size - 1) / max_xfer + 1;
        if (n_cmds > UINT16_MAX) {
            return make_error(UGDS_INVALID_VALUE);
        }
        entry.n_cmds = static_cast<uint16_t>(n_cmds);
    }

    size_t total_cmds = 0;
    for (unsigned i = 0; i < nr; ++i)
        total_cmds += bs->entries[base + i].n_cmds;

    // Phase 2: build sub-command list
    struct SubCmd {
        unsigned io_idx;
        uint64_t lba;
        size_t   page_start;
        size_t   chunk_size;
        nvm_dma_t* buf_dma;
    };

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
                buf_dma = it->second.dma;
                /* Hold in-flight reference until batch completions are drained
                 * or destroy finishes. Prevents Deregister from unmapping
                 * while batch commands retain PRPs from this buffer. */
                it->second.in_flight.fetch_add(1, std::memory_order_acq_rel);
            }
        }

        if (buf_dma == nullptr) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
            continue;
        }

        if ((static_cast<size_t>(entry.devPtr_offset) % page_size) != 0) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
            async_release_inflight_batch(entry.devPtr_base);
            continue;
        }
        buf_page_start = static_cast<size_t>(entry.devPtr_offset) / page_size;

        /* Overflow-safe bounds check */
        size_t batch_pages_needed = (entry.size - 1) / page_size + 1;
        if (batch_pages_needed > buf_dma->n_ioaddrs ||
            buf_page_start > buf_dma->n_ioaddrs - batch_pages_needed) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
            async_release_inflight_batch(entry.devPtr_base);
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

    // Phase 3: enqueue all NVMe commands to the single batch QP
    {
        std::lock_guard<std::mutex> qp_lock(qp.lock);

        for (auto& sc : work) {
            BatchIOEntry& entry = bs->entries[sc.io_idx];
            size_t n_pages = (sc.chunk_size + page_size - 1) / page_size;
            if (n_pages == 0) n_pages = 1;

            uint16_t prp_idx = UINT16_MAX;
            if (n_pages > 2) {
                int pidx = prp_pool_alloc_page(&bs->prp_pool);
                if (pidx < 0) {
                    /* Resource backpressure: commands already built must run
                     * before Submit can reuse their PRP-list pages. */
                    nvm_sq_submit(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    uint64_t spins = 0;
                    const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
                    while ((pidx = prp_pool_alloc_page(&bs->prp_pool)) < 0) {
                        if (drain_completions(qp, bs) == 0) {
                            if (++spins > max_spins) {
                                /* Release in_flight refs only for entries
                                 * that are still WAITING (ref acquired but
                                 * not yet submitted). PENDING entries have
                                 * commands in the SQ -- keep ref until drain.
                                 * COMPLETE entries were already released by
                                 * drain_completions. FAILED already done. */
                                for (unsigned i = 0; i < nr; ++i) {
                                    BatchIOEntry& e = bs->entries[base + i];
                                    if (e.status == UGDS_BATCH_WAITING) {
                                        async_release_inflight_batch(e.devPtr_base);
                                        e.status = UGDS_BATCH_FAILED;
                                        e.n_cmds = 0;
                                    }
                                }
                                bs->n_entries += nr;
                                return make_error(UGDS_INTERNAL_ERROR);
                            }
                            __builtin_ia32_pause();
                        } else {
                            spins = 0;
                        }
                    }
                }
                prp_idx = static_cast<uint16_t>(pidx);
            }

            uint16_t slot = static_cast<uint16_t>(
                qp.sq.tail.load(std::memory_order_relaxed) % qp.sq.qs);
            nvm_cmd_t* cmd = nvm_sq_enqueue(&qp.sq);

            if (cmd == nullptr) {
                /* Resource backpressure: expose the queued SQEs and reclaim
                 * completed SQ slots locally, regardless of completion mode. */
                nvm_sq_submit(&qp.sq);
                std::atomic_thread_fence(std::memory_order_seq_cst);

                uint64_t spins = 0;
                const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
                bool drained = false;
                while (!drained) {
                    if (drain_completions(qp, bs) > 0) {
                        drained = true;
                    } else if (++spins > max_spins) {
                        if (prp_idx != UINT16_MAX)
                            prp_pool_release_page(&bs->prp_pool, prp_idx);
                        /* Release in_flight refs only for WAITING entries. */
                        for (unsigned i = 0; i < nr; ++i) {
                            BatchIOEntry& e = bs->entries[base + i];
                            if (e.status == UGDS_BATCH_WAITING) {
                                async_release_inflight_batch(e.devPtr_base);
                                e.status = UGDS_BATCH_FAILED;
                                e.n_cmds = 0;
                            }
                        }
                        bs->n_entries += nr;
                        return make_error(UGDS_INTERNAL_ERROR);
                    } else {
                        __builtin_ia32_pause();
                    }
                }

                slot = static_cast<uint16_t>(
                    qp.sq.tail.load(std::memory_order_relaxed) % qp.sq.qs);
                cmd = nvm_sq_enqueue(&qp.sq);
                if (cmd == nullptr) {
                    if (prp_idx != UINT16_MAX)
                        prp_pool_release_page(&bs->prp_pool, prp_idx);
                    /* Release in_flight refs only for WAITING entries. */
                    for (unsigned i = 0; i < nr; ++i) {
                        BatchIOEntry& e = bs->entries[base + i];
                        if (e.status == UGDS_BATCH_WAITING) {
                            async_release_inflight_batch(e.devPtr_base);
                            e.status = UGDS_BATCH_FAILED;
                            e.n_cmds = 0;
                        }
                    }
                    bs->n_entries += nr;
                    return make_error(UGDS_INTERNAL_ERROR);
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

            entry.status = UGDS_BATCH_PENDING;
        }

        /* Expose all newly enqueued commands with one SQ doorbell write. */
        std::atomic_thread_fence(std::memory_order_seq_cst);
        nvm_sq_submit(&qp.sq);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    bs->n_entries += nr;
    return UGDS_OK;
}

static bool status_request_satisfied(unsigned min_events,
                                     unsigned max_events,
                                     unsigned n_ready)
{
    return min_events == 0 || n_ready >= min_events || n_ready >= max_events;
}

/* Drain hardware completions and copy newly completed logical batch entries.
 * The CQ lock protects queue head/doorbell state; the batch lock protects the
 * command map, entry state and event_returned flags. */
static bool collect_batch_events(BatchState* bs, IOQueuePair& qp,
                                 unsigned min_events, unsigned max_events,
                                 uGDSIOEvents_t* events, unsigned* n_ready)
{
    std::lock_guard<std::mutex> batch_lock(bs->lock);

    if (bs->in_flight > 0) {
        std::lock_guard<std::mutex> qp_lock(qp.lock);
        drain_completions(qp, bs);
    }

    for (unsigned i = 0;
         i < bs->n_entries && *n_ready < max_events; ++i) {
        BatchIOEntry& entry = bs->entries[i];
        if (entry.event_returned)
            continue;
        if (entry.status != UGDS_BATCH_COMPLETE &&
            entry.status != UGDS_BATCH_FAILED)
            continue;

        events[*n_ready].cookie = entry.cookie;
        events[*n_ready].status = entry.status;
        events[*n_ready].ret = entry.error_code;
        entry.event_returned = true;
        bs->n_events_read++;
        (*n_ready)++;
    }

    return status_request_satisfied(min_events, max_events, *n_ready);
}

static void make_deadline(const struct timespec& timeout,
                          struct timespec* deadline)
{
    clock_gettime(CLOCK_MONOTONIC, deadline);
    deadline->tv_sec += timeout.tv_sec;
    deadline->tv_nsec += timeout.tv_nsec;
    if (deadline->tv_nsec >= 1000000000L) {
        deadline->tv_sec += deadline->tv_nsec / 1000000000L;
        deadline->tv_nsec %= 1000000000L;
    }
}

/* Return false when the absolute deadline has expired. */
static bool remaining_time(const struct timespec* deadline,
                           struct timespec* remaining)
{
    if (deadline == nullptr)
        return true;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    remaining->tv_sec = deadline->tv_sec - now.tv_sec;
    remaining->tv_nsec = deadline->tv_nsec - now.tv_nsec;
    if (remaining->tv_nsec < 0) {
        remaining->tv_sec--;
        remaining->tv_nsec += 1000000000L;
    }
    return remaining->tv_sec >= 0;
}

static uGDSError_t get_status_polling(BatchState* bs, IOQueuePair& qp,
                                      unsigned min_events,
                                      unsigned max_events,
                                      uGDSIOEvents_t* events,
                                      const struct timespec* deadline,
                                      unsigned* nr)
{
    unsigned n_ready = 0;

    for (;;) {
        if (collect_batch_events(bs, qp, min_events, max_events,
                                 events, &n_ready)) {
            *nr = n_ready;
            return UGDS_OK;
        }

        struct timespec unused;
        if (!remaining_time(deadline, &unused)) {
            /* Close the timeout edge: a CQE may have arrived between the last
             * drain and the clock check. */
            collect_batch_events(bs, qp, min_events, max_events,
                                 events, &n_ready);
            *nr = n_ready;
            return UGDS_OK;
        }

        __builtin_ia32_pause();
    }
}

static uGDSError_t get_status_interrupt(BatchState* bs, IOQueuePair& qp,
                                        unsigned min_events,
                                        unsigned max_events,
                                        uGDSIOEvents_t* events,
                                        const struct timespec* deadline,
                                        unsigned* nr)
{
    unsigned n_ready = 0;

    for (;;) {
        /* Always inspect the CQ before blocking.  eventfd is only a wakeup
         * hint; the CQ phase bit remains the source of completion truth. */
        if (collect_batch_events(bs, qp, min_events, max_events,
                                 events, &n_ready)) {
            *nr = n_ready;
            return UGDS_OK;
        }

        struct timespec wait_time;
        struct timespec* wait_ptr = nullptr;
        if (deadline != nullptr) {
            if (!remaining_time(deadline, &wait_time)) {
                /* As in polling mode, perform a final CQ drain at timeout. */
                collect_batch_events(bs, qp, min_events, max_events,
                                     events, &n_ready);
                *nr = n_ready;
                return UGDS_OK;
            }
            wait_ptr = &wait_time;
        }

        struct pollfd pfd = {qp.irq_efd, POLLIN, 0};
        int pr = ppoll(&pfd, 1, wait_ptr, nullptr);
        if (pr < 0) {
            if (errno == EINTR)
                continue;
            return make_error(UGDS_INTERNAL_ERROR);
        }
        if (pr == 0)
            continue;
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
            return make_error(UGDS_INTERNAL_ERROR);
        if (!(pfd.revents & POLLIN))
            continue;

        /* Reading acknowledges all IRQs accumulated in eventfd.  The next
         * loop iteration drains every CQE currently ready as one burst. */
        uint64_t count;
        ssize_t ret = read(qp.irq_efd, &count, sizeof(count));
        if (ret == static_cast<ssize_t>(sizeof(count)))
            continue;
        if (ret < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
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

    struct timespec deadline;
    struct timespec* deadline_ptr = nullptr;
    if (timeout != nullptr) {
        make_deadline(*timeout, &deadline);
        deadline_ptr = &deadline;
    }

    /* A batch CQ has one consuming eventfd, so only one GetStatus call may
     * own its wait/collect cycle at a time.  Submit uses bs->lock separately
     * and can still run while an interrupt-mode caller sleeps here. */
    std::lock_guard<std::mutex> status_guard(bs->status_lock);

    if (qp.irq_efd < 0)
        return get_status_polling(bs, qp, min_nr, max_events, events,
                                  deadline_ptr, nr);

    return get_status_interrupt(bs, qp, min_nr, max_events, events,
                                deadline_ptr, nr);
}

extern "C" void uGDSBatchIODestroy(uGDSBatchHandle_t batch)
{
    if (batch == nullptr) return;

    BatchState* bs = static_cast<BatchState*>(batch);
    HandleState* hs = bs->hs;
    IOQueuePair& qp = hs->batch_qp->qp;

    // Drain remaining in-flight commands
    if (bs->in_flight > 0) {
        std::lock_guard<std::mutex> qp_lock(qp.lock);
        uint64_t spins = 0;
        const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
        while (bs->in_flight > 0) {
            if (drain_completions(qp, bs) == 0) {
                if (++spins > max_spins) break;
                __builtin_ia32_pause();
            } else {
                spins = 0;
            }
        }
    }

    /* Best-effort: release in-flight references for entries that were
     * submitted but never fully completed (e.g., submit phase-3 early
     * return). Check n_cmds_done < n_cmds rather than status alone.
     *
     * IMPORTANT: do NOT release refs for entries that have commands
     * still outstanding in the NVMe SQ (bs->in_flight > 0 after drain).
     * Those commands may still DMA. Leaking the refs (blocking
     * deregister) is strictly safer than freeing mappings while the
     * device may still access them. The caller should reset the
     * controller if truly stuck. */
    if (bs->in_flight == 0) {
        for (unsigned i = 0; i < bs->n_entries; ++i) {
            BatchIOEntry& entry = bs->entries[i];
            if (entry.n_cmds > 0 && entry.n_cmds_done < entry.n_cmds) {
                async_release_inflight_batch(entry.devPtr_base);
                entry.status = UGDS_BATCH_FAILED;
            }
        }
    } else {
        /* Commands still in flight after drain timeout.
         * The NVMe device may still DMA through the PRP list and CQ,
         * so we must NOT: free the PRP pool, release handle ref,
         * delete the batch state, or clear batch_active.
         *
         * Keeping batch_active=true prevents a new BatchIOSetUp from
         * reusing the same QP while old completions may still arrive.
         * Keeping the handle ref prevents HandleDeregister from freeing
         * the QP while DMA is in progress.
         *
         * This wedges the handle until the caller resets the controller.
         * BatchIODestroy is void so the caller cannot detect this --
         * the warning is the only signal. This is the safest option:
         * a wedged handle is strictly better than DMA-after-unmap. */
        fprintf(stderr, "uGDS: BatchIODestroy: %u commands still in "
                "flight after drain timeout -- handle is now wedged "
                "to prevent DMA-after-unmap. Controller reset required.\n",
                bs->in_flight);
        hs->wedged.store(true, std::memory_order_release);
        return;
    }

    prp_pool_destroy(bs);

    /* Mark batch inactive before releasing handle ref.
     * HandleDeregister waits on handle_in_flight==0, so we must not
     * touch hs after release. */
    hs->batch_active.store(false);

    /* Release handle reference acquired in BatchIOSetUp */
    handle_release(hs);

    delete bs;
}
