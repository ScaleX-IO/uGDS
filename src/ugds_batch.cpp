#include "ugds_internal.h"

#include <cstring>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <atomic>
#include <time.h>

static void cleanup_prp_pool(BatchState* bs)
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

    if (entry.n_cmds_done == entry.n_cmds) {
        if (entry.status != UGDS_BATCH_FAILED) {
            entry.status = UGDS_BATCH_COMPLETE;
        }
        entry.error_code = entry.bytes_done;
        bs->n_completed++;
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
    if (batch == nullptr || fh == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (nr == 0 || nr > UGDS_MAX_BATCH_IO_SIZE)
        return make_error(UGDS_INVALID_VALUE);

    HandleState* hs = static_cast<HandleState*>(fh);

    if (!hs->batch_qp)
        return make_error(UGDS_INTERNAL_ERROR);

    bool expected = false;
    if (!hs->batch_active.compare_exchange_strong(expected, true))
        return make_error(UGDS_INVALID_VALUE);

    auto bs = new (std::nothrow) BatchState();
    if (!bs) {
        hs->batch_active.store(false);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    bs->capacity = nr;
    bs->hs = hs;
    bs->entries.resize(nr);
    bs->cmd_map.resize(hs->batch_queue_depth);

    const size_t page_size = hs->ctrl->page_size;
    PRPPool& pool = bs->prp_pool;
    size_t pool_bytes = UGDS_PRP_POOL_PAGES * page_size;

    if (posix_memalign(&pool.buf, 4096, pool_bytes) != 0) {
        delete bs;
        hs->batch_active.store(false);
        return make_error(UGDS_INTERNAL_ERROR);
    }
    std::memset(pool.buf, 0, pool_bytes);

    int rc = nvm_dma_map_host(&pool.dma, hs->ctrl, pool.buf, pool_bytes);
    if (!nvm_ok(rc)) {
        free(pool.buf);
        pool.buf = nullptr;
        delete bs;
        hs->batch_active.store(false);
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

        size_t n_cmds = (p.size + max_xfer - 1) / max_xfer;
        if (n_cmds == 0) n_cmds = 1;
        entry.n_cmds = static_cast<uint16_t>(n_cmds);
    }

    // Phase 2: build sub-command list
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
            if (it != g_driver.buf_registry.end())
                buf_dma = it->second;
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
            continue;
        }
        buf_page_start = static_cast<size_t>(entry.devPtr_offset) / page_size;

        size_t total_pages_needed = buf_page_start +
            (entry.size + page_size - 1) / page_size;
        if (total_pages_needed > buf_dma->n_ioaddrs) {
            entry.status = UGDS_BATCH_FAILED;
            entry.error_code = -EINVAL;
            entry.n_cmds = 0;
            entry.n_cmds_done = 0;
            bs->n_completed++;
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
                int pidx = prp_pool_alloc(&bs->prp_pool);
                if (pidx < 0) {
                    nvm_sq_submit(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    uint64_t spins = 0;
                    const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
                    while ((pidx = prp_pool_alloc(&bs->prp_pool)) < 0) {
                        if (!drain_one_completion(qp, bs)) {
                            if (++spins > max_spins) {
                                bs->n_entries += nr;
                                return make_error(UGDS_INTERNAL_ERROR);
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
                        prp_pool_free(&bs->prp_pool, prp_idx);
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

        // Single doorbell for all commands
        std::atomic_thread_fence(std::memory_order_seq_cst);
        nvm_sq_submit(&qp.sq);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    bs->n_entries += nr;
    return UGDS_OK;
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
    IOQueuePair& qp = hs->batch_qp->qp;

    // Drain remaining in-flight commands
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

    cleanup_prp_pool(bs);
    hs->batch_active.store(false);
    delete bs;
}
