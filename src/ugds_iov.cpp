/*
 * Copyright (c) 2024, Guanyi Chen <felixlinker02@gmail.com>
 * Copyright (c) 2017, Jonas Markauss <jonassm@ifi.uio.no>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* uGDS SGL (scatter-gather) streaming engine and sync vectored API.
 *
 * Public APIs implemented here:
 *   - uGDSReadv / uGDSWritev       (sync vectored IO)
 *
 * Internal primitives implemented here (consumed by ugds_batch.cpp and
 * ugds_async.cpp):
 *   - SglRefOwner                  (fixed-capacity in-flight ref owner)
 *   - SglWindowCursor / SglPageCursor / sgl_count_windows_analytic
 *   - do_iov_engine                (shared streaming windowed IO engine)
 *
 * Implements the pure streaming primitives (SglPageCursor,
 * SglWindowCursor, analytic counter), the fixed-capacity reference
 * owner (SglRefOwner), the shared windowed IO engine (do_iov_engine),
 * and the public sync vectored entry points (uGDSReadv/uGDSWritev).
 *
 * Design constraints honored here:
 *  - SglWindowCursor yields one CmdWindow at a time into caller-local
 *    storage; no array of windows is ever materialized.
 *  - SglRefOwner is fixed-capacity for UGDS_IOV_MAX and never allocates;
 *    it is noexcept move-only so timeout parking is a bounded array move.
 *  - HandleOpGuard owns the handle-operation reference from immediately
 *    after handle_lookup through engine return, closing the OOM-strand
 *    window.
 *  - Exact registered-length bounds and controller affinity are checked
 *    under g_driver.lock before the first cursor.next().
 */

#include "ugds_internal.h"

#include <cstring>
#include <cstdio>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <new>

#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(((size_t)1 << (sizeof(ssize_t) * 8 - 1)) - 1))
#endif

/* ========================================================================
 * SglRefOwner implementation
 * ======================================================================== */

SglRefOwner::SglRefOwner(SglRefOwner&& other) noexcept
    : bases_(other.bases_), size_(other.size_)
{
    other.size_ = 0;
}

SglRefOwner& SglRefOwner::operator=(SglRefOwner&& other) noexcept
{
    if (this != &other) {
        /* Release anything we currently hold before overwriting. */
        release();
        bases_ = other.bases_;
        size_  = other.size_;
        other.size_ = 0;
    }
    return *this;
}

SglRefOwner::~SglRefOwner() noexcept
{
    release();
}

void SglRefOwner::release() noexcept
{
    if (size_ == 0) return;
    std::lock_guard<std::mutex> g(g_driver.lock);
    for (uint16_t i = 0; i < size_; ++i) {
        auto it = g_driver.buf_registry.find(bases_[i]);
        if (it != g_driver.buf_registry.end())
            it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }
    size_ = 0;
}

/* Acquire in-flight references for all segments under one g_driver.lock hold.
 * All-or-nothing: on any failure, rolls back [0..k-1].  Returns 0 on success
 * or -EINVAL.  If seg_views_out is non-null, fills the resolved SegView[]. */
int SglRefOwner::acquire(const uGDSIoSegment_t* segs, uint32_t nr,
                         const nvm_ctrl_t* hs_ctrl,
                         size_t page_size,
                         SegView* seg_views_out)
{
    std::lock_guard<std::mutex> g(g_driver.lock);

    for (uint32_t k = 0; k < nr; ++k) {
        const void* base = segs[k].base;
        auto it = g_driver.buf_registry.find(base);
        if (it == g_driver.buf_registry.end()) {
            /* Rollback [0..k-1] */
            for (uint32_t j = 0; j < k; ++j)
                g_driver.buf_registry.find(bases_[j])->second.in_flight
                    .fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }

        DriverState::BufEntry& entry = it->second;

        /* Controller affinity check: mapping controller must match the
         * submitting handle's controller. */
        if (entry.map_ctrl != hs_ctrl) {
            for (uint32_t j = 0; j < k; ++j)
                g_driver.buf_registry.find(bases_[j])->second.in_flight
                    .fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }

        /* Exact-length bounds, subtraction form.  offset and size were
         * already value-checked MPS/block-aligned by the caller; here we
         * enforce the registered-range bound. */
        const uint64_t off = static_cast<uint64_t>(segs[k].offset);
        const uint64_t sz  = static_cast<uint64_t>(segs[k].size);
        if (off > entry.length || sz > entry.length - off) {
            for (uint32_t j = 0; j < k; ++j)
                g_driver.buf_registry.find(bases_[j])->second.in_flight
                    .fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }

        /* Acquire the reference and record the base. */
        entry.in_flight.fetch_add(1, std::memory_order_acq_rel);
        bases_[k] = base;

        /* Fill the resolved SegView if requested. */
        if (seg_views_out != nullptr) {
            SegView& sv = seg_views_out[k];
            sv.dma               = entry.dma;
            sv.base              = base;
            sv.registered_length = entry.length;
            sv.backend           = entry.backend;
            sv.page_start        = off / page_size;
            sv.size              = segs[k].size;
        }
    }
    size_ = static_cast<uint16_t>(nr);
    return 0;
}

/* Identity-only acquire for the async path: skips the exact-length bound
 * because offset/size are late-bound and not yet observable at enqueue.
 * Fills only dma/base/registered_length/backend in seg_views_out; geometry
 * is completed by the callback. */
int SglRefOwner::acquire_identity_only(const uGDSIoSegment_t* segs, uint32_t nr,
                                       const nvm_ctrl_t* hs_ctrl,
                                       SegView* seg_views_out)
{
    std::lock_guard<std::mutex> g(g_driver.lock);

    for (uint32_t k = 0; k < nr; ++k) {
        const void* base = segs[k].base;
        auto it = g_driver.buf_registry.find(base);
        if (it == g_driver.buf_registry.end()) {
            for (uint32_t j = 0; j < k; ++j)
                g_driver.buf_registry.find(bases_[j])->second.in_flight
                    .fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }

        DriverState::BufEntry& entry = it->second;

        /* Controller affinity check. */
        if (entry.map_ctrl != hs_ctrl) {
            for (uint32_t j = 0; j < k; ++j)
                g_driver.buf_registry.find(bases_[j])->second.in_flight
                    .fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }

        /* Acquire the reference and record the base. */
        entry.in_flight.fetch_add(1, std::memory_order_acq_rel);
        bases_[k] = base;

        /* Identity-only SegView: geometry (page_start/size) left zero for
         * the callback to fill after late binding. */
        if (seg_views_out != nullptr) {
            SegView& sv = seg_views_out[k];
            sv.dma               = entry.dma;
            sv.base              = base;
            sv.registered_length = entry.length;
            sv.backend           = entry.backend;
            sv.page_start        = 0;
            sv.size              = 0;
        }
    }
    size_ = static_cast<uint16_t>(nr);
    return 0;
}

/* ========================================================================
 * SglWindowCursor
 * ======================================================================== */

/* Compute SU = lcm(MPS, block_size).  Both are powers of two (MPS = 2^(12+n),
 * LBADS >= 9), so lcm = max; the code computes the power-of-two case directly
 * and falls back to lcm defensively. */
static size_t compute_split_unit(size_t mps, size_t block_size) noexcept
{
    /* Power-of-two check: if both have exactly one bit set, lcm = max. */
    if (mps > 0 && block_size > 0 &&
        (mps & (mps - 1)) == 0 &&
        (block_size & (block_size - 1)) == 0) {
        return std::max(mps, block_size);
    }
    /* Defensive gcd/lcm for non-power-of-two (should not occur per NVMe). */
    size_t a = mps, b = block_size;
    while (b != 0) { size_t t = a % b; a = b; b = t; }
    return (mps / a) * block_size;
}

/* Initialize a cursor for a resolved SegView array.  Returns false
 * (reject) if window_cap, page_size, block_size, or nr_segs is zero.
 * The cursor holds no storage of its own; 'segs' must outlive the
 * cursor walk. */
bool sgl_cursor_init(SglWindowCursor& c, const SegView* segs, uint32_t nr_segs,
                     size_t window_cap, size_t page_size, size_t block_size)
{
    if (window_cap == 0 || page_size == 0 || block_size == 0 || nr_segs == 0)
        return false;
    c.segs       = segs;
    c.nr_segs    = nr_segs;
    c.seg_idx    = 0;
    c.bytes_in_seg = 0;
    c.window_cap = window_cap;
    c.page_size  = page_size;
    c.block_size = block_size;
    c.split_unit = compute_split_unit(page_size, block_size);
    return true;
}

/* Emit at most one window into out.  Returns true if a window was produced,
 * false when the SGL is exhausted.  Implements the capacity / PRP-tail /
 * end-of-SGL rules without materializing a window array. */
bool sgl_cursor_next(SglWindowCursor& c, CmdWindow& out) noexcept
{
    if (c.seg_idx >= c.nr_segs)
        return false;

    out.first_seg          = c.seg_idx;
    out.first_seg_page_off = c.bytes_in_seg / c.page_size;
    out.n_segs             = 0;
    out.bytes              = 0;
    out.n_pages            = 0;

    while (c.seg_idx < c.nr_segs) {
        const SegView& seg = c.segs[c.seg_idx];
        size_t rem  = seg.size - c.bytes_in_seg;
        size_t take = std::min(rem, c.window_cap - out.bytes);

        /* Capacity rule: if this take splits mid-segment, round down to an
         * SU boundary so the continuation starts page-aligned. */
        if (take < rem) {
            take = (take / c.split_unit) * c.split_unit;
            /* Safety: window contents before a split are SU-multiples up to
             * here, window_cap is an SU multiple, and a non-closed window
             * has window_cap - out.bytes >= SU, so take >= SU > 0. */
        }

        size_t slice_pages = (take + c.page_size - 1) / c.page_size;
        out.n_segs++;
        out.bytes   += take;
        out.n_pages += slice_pages;
        c.bytes_in_seg += take;

        bool segment_ended = (c.bytes_in_seg == seg.size);
        if (segment_ended) {
            /* PRP tail rule: a segment whose size is not page-multiple
             * must be the window's last slice. */
            bool non_page_tail = (seg.size % c.page_size != 0);
            c.seg_idx++;
            c.bytes_in_seg = 0;
            if (non_page_tail) {
                return true;  /* PRP tail */
            }
        }

        /* Capacity rule */
        if (out.bytes >= c.window_cap)
            return true;
    }

    /* End of SGL: return whatever was accumulated. */
    return true;
}

/* ========================================================================
 * sgl_count_windows_analytic
 * O(nr_segs) exact count via room/cap arithmetic.
 * ======================================================================== */

bool sgl_count_windows_analytic(const uGDSIoSegment_t* segs,
                                uint32_t nr_segs,
                                size_t window_cap, size_t page_size,
                                size_t block_size,
                                uint32_t* out) noexcept
{
    if (window_cap == 0 || page_size == 0 || block_size == 0 || nr_segs == 0)
        return false;

    /* SU is used by the cursor to round capacity-rule split points; the
     * analytic counter uses quotient/remainder against window_cap, which
     * the caller already guaranteed is an SU multiple (so full windows
     * need no rounding).  We keep SU here only as a debug sanity
     * reference. */
    const size_t SU = compute_split_unit(page_size, block_size);
    (void)SU;

    uint64_t count = 0;
    uint64_t open  = 0;  /* bytes accumulated in the current open window */

    for (uint32_t i = 0; i < nr_segs; ++i) {
        uint64_t s = segs[i].size;

        /* Absorb into the open window if one exists. */
        if (open != 0) {
            uint64_t room = window_cap - open;
            if (s < room) {
                open += s;
                s = 0;
            } else {
                /* s >= room: the open window fills to capacity.
                 * But we must respect the capacity-rule mid-segment
                 * rounding: the actual take is round_down(room, SU).
                 * Since window contents before a split are SU-multiples
                 * and window_cap is an SU multiple, room itself is an SU
                 * multiple when open is an SU multiple (which it is by
                 * induction).  So take == room exactly, no rounding
                 * loss. */
                s -= room;
                count++;
                open = 0;
            }
        }

        if (s != 0) {
            /* All full capacity-rule windows at once.  Each full window
             * takes window_cap bytes (an SU multiple), so no rounding. */
            count += s / window_cap;
            open = s % window_cap;
        }

        /* PRP tail rule: if this segment's size is not page-multiple, it closes the
         * window. */
        if (segs[i].size % page_size != 0) {
            /* open must be nonzero here: the segment just contributed
             * remainder bytes and was not absorbed entirely into a prior
             * open window. */
            if (open != 0) {
                count++;
                open = 0;
            }
        }
    }

    /* End of SGL: trailing open window */
    if (open != 0) count++;

    if (count > UINT32_MAX)
        return false;

    *out = static_cast<uint32_t>(count);
    return true;
}

/* ========================================================================
 * SglPageCursor
 * ======================================================================== */

/* Initialize a page cursor for one CmdWindow.  The caller then calls
 * sgl_page_cursor_next() exactly CmdWindow::n_pages times to walk the
 * MPS-granular bus addresses across segment boundaries. */
void sgl_page_cursor_init(SglPageCursor& c, const SegView* segs,
                          uint32_t first_seg, size_t first_seg_page_off,
                          uint32_t n_segs, size_t n_pages) noexcept
{
    c.segs            = segs;
    c.abs_seg         = first_seg;
    c.end_seg         = first_seg + n_segs;
    c.page_in_seg     = segs[first_seg].page_start + first_seg_page_off;
    /* Compute the pages in the first slice.  For the first segment the slice
     * may start mid-segment (first_seg_page_off > 0).  The total pages in
     * the slice is ceil(slice_bytes / MPS).  We don't have slice_bytes
     * directly, but we know:
     *  - For the last segment in the window, the slice may be partial.
     *  - For non-last segments (or single-segment full), the slice covers
     *    from page_in_seg to the end of the segment's registered pages,
     *    minus what later windows consume (but within this window it's the
     *    full remainder).
     *
     * Since the caller will call next() exactly n_pages times total, and
     * each segment boundary is page-aligned (alignment + cursor rules), we
     * can lazily compute slice pages as: for a given segment, the slice
     * runs from page_in_seg to min(seg_end_page, page_in_seg + remaining).
     * We track pages_remaining_total and cap each segment's contribution. */
    c.slice_pages_left = n_pages;  /* total; will be reduced per-segment */
    (void)first_seg_page_off;
}

/* Return the current segment's ioaddr and advance the cursor across
 * segment boundaries.  Total calls bounded by CmdWindow::n_pages.
 * Returns 0 only on misuse (caller overran n_pages). */
uint64_t sgl_page_cursor_next(SglPageCursor& c) noexcept
{
    /* Walk the segment slices in order, emitting MPS-granular bus addresses.
     * slice_pages_left tracks the total pages remaining across all slices
     * in this window; each segment boundary naturally caps a slice. */
    while (c.abs_seg < c.end_seg && c.slice_pages_left > 0) {
        const SegView& seg = c.segs[c.abs_seg];
        /* The slice for this segment runs from page_in_seg up to either the
         * segment's page boundary or slice_pages_left exhaustion.  Since all
         * slices except possibly the last are page-multiple (alignment +
         * cursor rules), and the last slice is capped by slice_pages_left,
         * this walk is correct. */
        size_t seg_end_page = seg.page_start +
            (seg.size + seg.dma->page_size - 1) / seg.dma->page_size;
        if (c.page_in_seg < seg_end_page) {
            uint64_t addr = seg.dma->ioaddrs[c.page_in_seg];
            c.page_in_seg++;
            c.slice_pages_left--;
            return addr;
        }
        /* Advance to the next segment in the window. */
        c.abs_seg++;
        if (c.abs_seg < c.end_seg) {
            c.page_in_seg = c.segs[c.abs_seg].page_start;
        }
    }
    return 0;  /* safety; should not happen with correct use */
}

/* ========================================================================
 * do_iov_engine
 * Shared streaming windowed IO engine.
 * ======================================================================== */

IovEngineResult do_iov_engine(HandleState* hs, SegView* segs, uint32_t nr_segs,
                              off_t file_offset, uint8_t opcode,
                              SglRefOwner* owner, nvm_dma_t* transient)
{
    IovEngineResult result{0, false};

    const size_t page_size  = hs->ctrl->page_size;
    const size_t block_size = hs->block_size;

    /* Compute max_xfer (parity with the original scalar path). */
    const size_t prp_capacity = page_size / sizeof(uint64_t);
    size_t max_xfer = hs->max_transfer_size;
    if (max_xfer == 0)
        max_xfer = UGDS_DEFAULT_MAX_TRANSFER_SIZE;
    if (max_xfer < page_size)
        max_xfer = page_size;
    size_t prp_max = (prp_capacity + 1) * page_size;
    if (max_xfer > prp_max)
        max_xfer = prp_max;

    /* Compute SU and window_cap = round_down(max_xfer, SU). */
    const size_t SU = compute_split_unit(page_size, block_size);
    size_t window_cap = (max_xfer / SU) * SU;
    if (window_cap == 0) {
        result.ret = -EINVAL;
        return result;
    }

    /* Initialize the window cursor. */
    SglWindowCursor cursor;
    if (!sgl_cursor_init(cursor, segs, nr_segs, window_cap, page_size, block_size)) {
        result.ret = -EINVAL;
        return result;
    }

    const uint64_t start_lba = static_cast<uint64_t>(file_offset) / block_size;

    /* QP selection: round-robin, one QP per IO (unchanged concurrency story). */
    const uint16_t qp_idx = static_cast<uint16_t>(
        hs->rr_counter.fetch_add(1) % hs->num_qps);
    IOQueuePair& qp = *hs->qps[qp_idx];

    ssize_t bytes_done = 0;
    uint64_t current_lba = start_lba;

    {
        std::lock_guard<std::mutex> qp_lock(qp.lock);

        /* Re-check wedged after acquiring QP lock. */
        if (hs->wedged.load(std::memory_order_acquire)) {
            result.ret = -EBADF;
            return result;
        }

        CmdWindow window;
        while (sgl_cursor_next(cursor, window)) {
            /* Enqueue a command, draining the CQ if the SQ is full. */
            nvm_cmd_t* cmd = nullptr;
            while ((cmd = nvm_sq_enqueue(&qp.sq)) == nullptr) {
                nvm_cpl_t* drain = wait_for_completion(hs, qp);
                if (drain == nullptr) {
                    /* Timeout during drain.  Park resources immediately
                     * while qp.lock is held. */
                    result.ret = -EIO;
                    result.timed_out = true;
                    hs->wedged.store(true, std::memory_order_release);
                    if (transient != nullptr) {
                        qp.timeout_dma = transient;
                        transient = nullptr;
                    } else if (owner != nullptr && !owner->empty()) {
                        qp.timeout_refs = std::move(*owner);
                    }
                    return result;
                }
                uint16_t st = UGDS_CPL_SCT_SC(drain);
                nvm_sq_update(&qp.sq);
                sgl_publish_prp();
                nvm_cq_update(&qp.cq);
                if (st != 0) {
                    result.ret = -EIO;
                    return result;
                }
            }

            memset(cmd, 0, sizeof(nvm_cmd_t));
            uint16_t cid = NVM_DEFAULT_CID(&qp.sq);
            nvm_cmd_header(cmd, cid, opcode, hs->ns_id);

            /* Build PRP using SglPageCursor. */
            SglPageCursor pc;
            sgl_page_cursor_init(pc, segs, window.first_seg,
                                 window.first_seg_page_off,
                                 window.n_segs, window.n_pages);

            if (window.n_pages == 1) {
                uint64_t prp1 = sgl_page_cursor_next(pc);
                nvm_cmd_data_ptr(cmd, prp1, 0);
            } else if (window.n_pages == 2) {
                uint64_t prp1 = sgl_page_cursor_next(pc);
                uint64_t prp2 = sgl_page_cursor_next(pc);
                nvm_cmd_data_ptr(cmd, prp1, prp2);
            } else {
                volatile uint64_t* prp_list =
                    reinterpret_cast<volatile uint64_t*>(qp.prp_dma->vaddr);
                uint64_t prp1 = sgl_page_cursor_next(pc);
                for (size_t i = 1; i < window.n_pages; ++i)
                    prp_list[i - 1] = sgl_page_cursor_next(pc);
                sgl_publish_prp();
                nvm_cmd_data_ptr(cmd, prp1, qp.prp_dma->ioaddrs[0]);
            }

            size_t n_blocks = window.bytes / block_size;
            nvm_cmd_rw_blks(cmd, current_lba, static_cast<uint16_t>(n_blocks));

            sgl_publish_prp();
            nvm_sq_submit(&qp.sq);
            sgl_publish_prp();

            nvm_cpl_t* cpl = wait_for_completion(hs, qp);
            if (cpl == nullptr) {
                /* Timeout: park resources immediately while qp.lock is held. */
                result.ret = -EIO;
                result.timed_out = true;
                hs->wedged.store(true, std::memory_order_release);
                if (transient != nullptr) {
                    qp.timeout_dma = transient;
                    transient = nullptr;
                } else if (owner != nullptr && !owner->empty()) {
                    qp.timeout_refs = std::move(*owner);
                }
                return result;
            }

            uint16_t status = UGDS_CPL_SCT_SC(cpl);
            nvm_sq_update(&qp.sq);
            sgl_publish_prp();
            nvm_cq_update(&qp.cq);

            if (status != 0) {
                result.ret = -EIO;
                return result;
            }

            bytes_done += static_cast<ssize_t>(window.bytes);
            current_lba += n_blocks;
        }

        result.ret = bytes_done;
    }

    return result;
}

/* ========================================================================
 * uGDSReadv / uGDSWritev
 * ======================================================================== */

/* Common validation + engine dispatch for vectored read/write.
 *
 * Performs, in order:
 *   1. Malformed-call checks (null segs, nr_segs == 0, nr_segs >
 *      UGDS_IOV_MAX) per the value matrix.
 *   2. handle_lookup + HandleOpGuard so every early return releases
 *      the handle-operation reference exactly once.
 *   3. Per-segment value validation (base/size nonzero, offset >= 0,
 *      offset MPS-aligned, size block-multiple, overflow-safe total).
 *   4. file_offset alignment and window_cap == 0 rejection.
 *   5. SglRefOwner::acquire under g_driver.lock (registration +
 *      controller affinity + exact-length bound, all-or-nothing).
 *   6. do_iov_engine dispatch.
 *   7. owner.release() (no-op if the engine parked it on timeout) and
 *      handle_guard.release().
 *
 * 'opcode' is NVM_IO_READ or NVM_IO_WRITE.  Returns total bytes or
 * -errno. */
static ssize_t do_readv_writev(uGDSHandle_t fh, const uGDSIoSegment_t* segs,
                               unsigned nr_segs, off_t file_offset,
                               uint8_t opcode)
{
    /* --- Malformed-call checks --- */
    if (segs == nullptr || nr_segs == 0)
        return -EINVAL;
    if (nr_segs > UGDS_IOV_MAX)
        return -EINVAL;

    /* handle_lookup + HandleOpGuard */
    std::shared_ptr<HandleState> hs_sp;
    HandleState* hs = handle_lookup(fh, &hs_sp);
    if (!hs)
        return -EBADF;
    HandleOpGuard handle_guard(hs);  /* owns handle_in_flight from here */

    const size_t page_size  = hs->ctrl->page_size;
    const size_t block_size = hs->block_size;

    if (page_size == 0 || block_size == 0) {
        return -EINVAL;
    }

    /* --- Per-segment value validation --- */
    uint64_t total_size = 0;
    for (unsigned i = 0; i < nr_segs; ++i) {
        if (segs[i].base == nullptr || segs[i].size == 0)
            return -EINVAL;
        if (segs[i].offset < 0)
            return -EINVAL;
        if ((static_cast<size_t>(segs[i].offset) % page_size) != 0)
            return -EINVAL;
        if ((segs[i].size % block_size) != 0)
            return -EINVAL;

        /* Overflow-safe total accumulation. Guard order matters:
         * total_size is already known to be <= SSIZE_MAX, so
         * SSIZE_MAX - total_size cannot underflow. */
        uint64_t seg_size = static_cast<uint64_t>(segs[i].size);
        if (seg_size > static_cast<uint64_t>(SSIZE_MAX) - total_size)
            return -EINVAL;  /* total overflow */
        total_size += seg_size;
    }

    /* file_offset alignment. */
    if (file_offset < 0 ||
        (static_cast<size_t>(file_offset) % block_size) != 0)
        return -EINVAL;

    /* window_cap == 0 check (block > max_xfer). */
    const size_t prp_capacity = page_size / sizeof(uint64_t);
    size_t max_xfer = hs->max_transfer_size;
    if (max_xfer == 0) max_xfer = UGDS_DEFAULT_MAX_TRANSFER_SIZE;
    if (max_xfer < page_size) max_xfer = page_size;
    size_t prp_max = (prp_capacity + 1) * page_size;
    if (max_xfer > prp_max) max_xfer = prp_max;
    const size_t SU = compute_split_unit(page_size, block_size);
    size_t window_cap = (max_xfer / SU) * SU;
    if (window_cap == 0)
        return -EINVAL;

    /* --- SegView storage: stack for nr <= 4, heap otherwise --- */
    /* HandleOpGuard protects the handle reference across this allocation,
     * so a bad_alloc here cannot strand handle_in_flight. */
    SegView stack_views[4];
    std::vector<SegView> heap_views;
    SegView* views = nullptr;

    if (nr_segs <= 4) {
        views = stack_views;
    } else {
        try {
            heap_views.resize(nr_segs);
        } catch (const std::bad_alloc&) {
            return -ENOMEM;
        }
        views = heap_views.data();
    }

    /* --- Acquire in-flight refs + resolve segments (all-or-nothing) --- */
    SglRefOwner owner;
    int rc = owner.acquire(segs, static_cast<uint32_t>(nr_segs),
                           hs->ctrl, page_size, views);
    if (rc != 0) {
        return rc;  /* -EINVAL: unregistered / bounds / affinity */
    }

    /* --- Engine dispatch --- */
    IovEngineResult result = do_iov_engine(hs, views,
                                           static_cast<uint32_t>(nr_segs),
                                           file_offset, opcode,
                                           &owner, nullptr);

    /* --- Cleanup --- */
    /* owner.release() is a no-op if the engine parked it on timeout. */
    owner.release();
    /* handle_guard.release() disarms the destructor and calls
     * handle_release exactly once. */
    handle_guard.release();

    if (result.timed_out) {
        fprintf(stderr, "uGDS: vectored I/O timeout -- handle wedged. "
                "Controller reset required.\n");
    }

    return result.ret;
}

extern "C" ssize_t uGDSReadv(uGDSHandle_t fh, const uGDSIoSegment_t* segs,
                               unsigned nr_segs, off_t file_offset)
{
    try {
        return do_readv_writev(fh, segs, nr_segs, file_offset, NVM_IO_READ);
    } catch (...) {
        return -EIO;
    }
}

extern "C" ssize_t uGDSWritev(uGDSHandle_t fh, const uGDSIoSegment_t* segs,
                                unsigned nr_segs, off_t file_offset)
{
    try {
        return do_readv_writev(fh, segs, nr_segs, file_offset, NVM_IO_WRITE);
    } catch (...) {
        return -EIO;
    }
}

/* The async vectored entry points (uGDSReadvAsync / uGDSWritevAsync) are
 * implemented in ugds_async.cpp. */
