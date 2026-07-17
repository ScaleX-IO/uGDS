#include "ugds_internal.h"

#include <cstring>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <poll.h>
#include <unistd.h>
#include <time.h>

static nvm_cpl_t* wait_for_completion(HandleState* hs, IOQueuePair& qp)
{
    if (qp.irq_efd < 0) {
        nvm_cpl_t* cpl = nullptr;
        uint64_t spins = 0;
        const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
        while ((cpl = nvm_cq_dequeue(&qp.cq)) == nullptr) {
            if (++spins > max_spins) break;
            __builtin_ia32_pause();
        }
        return cpl;
    }

    // Absolute deadline prevents EINTR restarts from stretching total wait.
    const long timeout_ms = (long)hs->ctrl->timeout;
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }

    for (;;) {
        // Poll CQ before blocking to prevent lost wakeups.
        nvm_cpl_t* cpl = nvm_cq_dequeue(&qp.cq);
        if (cpl != nullptr) return cpl;

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        struct timespec ts;
        ts.tv_sec = deadline.tv_sec - now.tv_sec;
        ts.tv_nsec = deadline.tv_nsec - now.tv_nsec;
        if (ts.tv_nsec < 0) {
            ts.tv_sec -= 1;
            ts.tv_nsec += 1000000000L;
        }
        if (ts.tv_sec < 0)
            return nvm_cq_dequeue(&qp.cq);

        struct pollfd pfd = {qp.irq_efd, POLLIN, 0};
        int pr = ppoll(&pfd, 1, &ts, nullptr);
        if (pr < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "uGDS: ppoll(eventfd=%d) failed: %s\n",
                    qp.irq_efd, strerror(errno));
            return nullptr;
        }
        if (pr == 0)
            return nvm_cq_dequeue(&qp.cq);

        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            fprintf(stderr, "uGDS: eventfd=%d invalid, revents=0x%x\n",
                    qp.irq_efd, pfd.revents);
            return nullptr;
        }
        if (!(pfd.revents & POLLIN))
            continue;

        uint64_t cnt;
        ssize_t r = read(qp.irq_efd, &cnt, sizeof(cnt));
        if (r == (ssize_t)sizeof(cnt))
            continue;
        if (r < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
        fprintf(stderr, "uGDS: read(eventfd=%d) failed: %s\n",
                qp.irq_efd, r < 0 ? strerror(errno) : "short read");
        return nullptr;
    }
}

ssize_t do_io_internal(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                       off_t file_offset, off_t bufPtr_offset, uint8_t opcode)
{
    HandleState* hs = static_cast<HandleState*>(fh);

    if (hs == nullptr || bufPtr_base == nullptr || size == 0) {
        return -EINVAL;
    }
    if (file_offset < 0 || bufPtr_offset < 0) {
        return -EINVAL;
    }
    if (hs->block_size == 0) {
        return -EINVAL;
    }
    if ((static_cast<size_t>(file_offset) % hs->block_size) != 0 ||
        (size % hs->block_size) != 0) {
        return -EINVAL;
    }

    const size_t page_size = hs->ctrl->page_size;
    if (page_size == 0) {
        return -EINVAL;
    }

    nvm_dma_t* buf_dma = nullptr;
    bool on_the_fly = false;
    size_t buf_page_start = 0;

    {
        std::lock_guard<std::mutex> drv_lock(g_driver.lock);
        auto it = g_driver.buf_registry.find(bufPtr_base);
        if (it != g_driver.buf_registry.end()) {
            buf_dma = it->second;
        }
    }

    if (buf_dma != nullptr) {
        if ((static_cast<size_t>(bufPtr_offset) % page_size) != 0) {
            return -EINVAL;
        }
        buf_page_start = static_cast<size_t>(bufPtr_offset) / page_size;
        size_t total_pages_needed = buf_page_start + (size + page_size - 1) / page_size;
        if (total_pages_needed > buf_dma->n_ioaddrs) {
            return -EINVAL;
        }
    } else {
        void* map_ptr = static_cast<uint8_t*>(bufPtr_base) + bufPtr_offset;
        // Reject non-64KB-aligned on-the-fly buffers (kernel driver silently rounds down)
        if ((reinterpret_cast<uintptr_t>(map_ptr) & ((1UL << 16) - 1)) != 0) {
            return -EINVAL;
        }
        int rc = nvm_dma_map_device(&buf_dma, hs->ctrl, map_ptr, size);
        if (rc != 0 || buf_dma == nullptr) {
            return -EIO;
        }
        on_the_fly = true;
        buf_page_start = 0;
    }

    const uint64_t start_lba = static_cast<uint64_t>(file_offset) / hs->block_size;

    const size_t prp_capacity = page_size / sizeof(uint64_t);

    size_t max_xfer = hs->max_transfer_size;
    if (max_xfer == 0) {
        max_xfer = UGDS_DEFAULT_MAX_TRANSFER_SIZE;
    }
    if (max_xfer < page_size) {
        max_xfer = page_size;
    }
    // Cap by PRP list capacity: PRP1 + prp_capacity entries → (prp_capacity + 1) pages
    size_t prp_max = (prp_capacity + 1) * page_size;
    if (max_xfer > prp_max) {
        max_xfer = prp_max;
    }

    const uint16_t qp_idx = static_cast<uint16_t>(
        hs->rr_counter.fetch_add(1) % hs->num_qps);
    IOQueuePair& qp = *hs->qps[qp_idx];

    ssize_t result = 0;
    size_t bytes_done = 0;
    uint64_t current_lba = start_lba;
    size_t current_page = buf_page_start;

    {
        std::lock_guard<std::mutex> qp_lock(qp.lock);

        while (bytes_done < size) {
            size_t remaining = size - bytes_done;
            size_t chunk_size = std::min(remaining, max_xfer);

            chunk_size = (chunk_size / hs->block_size) * hs->block_size;
            if (chunk_size == 0) {
                result = -EINVAL;
                goto out;
            }

            size_t n_blocks = chunk_size / hs->block_size;
            size_t n_pages = (chunk_size + page_size - 1) / page_size;
            if (n_pages == 0) n_pages = 1;

            nvm_cmd_t* cmd = nullptr;
            while ((cmd = nvm_sq_enqueue(&qp.sq)) == nullptr) {
                nvm_cpl_t* drain = wait_for_completion(hs, qp);
                if (drain == nullptr) {
                    result = -EIO;
                    goto out;
                }
                uint16_t st = UGDS_CPL_SCT_SC(drain);
                nvm_sq_update(&qp.sq);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                nvm_cq_update(&qp.cq);
                if (st != 0) {
                    result = -EIO;
                    goto out;
                }
            }

            memset(cmd, 0, sizeof(nvm_cmd_t));
            uint16_t cid = NVM_DEFAULT_CID(&qp.sq);
            nvm_cmd_header(cmd, cid, opcode, hs->ns_id);

            if (n_pages == 1) {
                nvm_cmd_data_ptr(cmd, buf_dma->ioaddrs[current_page], 0);
            } else if (n_pages == 2) {
                nvm_cmd_data_ptr(cmd,
                                 buf_dma->ioaddrs[current_page],
                                 buf_dma->ioaddrs[current_page + 1]);
            } else {
                volatile uint64_t* prp_list =
                    reinterpret_cast<volatile uint64_t*>(qp.prp_dma->vaddr);
                for (size_t i = 1; i < n_pages; ++i) {
                    prp_list[i - 1] = buf_dma->ioaddrs[current_page + i];
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                nvm_cmd_data_ptr(cmd,
                                 buf_dma->ioaddrs[current_page],
                                 qp.prp_dma->ioaddrs[0]);
            }

            nvm_cmd_rw_blks(cmd, current_lba, static_cast<uint16_t>(n_blocks));

            std::atomic_thread_fence(std::memory_order_seq_cst);
            nvm_sq_submit(&qp.sq);
            std::atomic_thread_fence(std::memory_order_seq_cst);

            nvm_cpl_t* cpl = wait_for_completion(hs, qp);
            if (cpl == nullptr) {
                result = -EIO;
                goto out;
            }

            uint16_t status = UGDS_CPL_SCT_SC(cpl);

            nvm_sq_update(&qp.sq);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            nvm_cq_update(&qp.cq);

            if (status != 0) {
                result = -EIO;
                goto out;
            }

            bytes_done += chunk_size;
            current_lba += n_blocks;
            current_page += n_pages;
        }

        result = static_cast<ssize_t>(bytes_done);
    out:;
    }

    if (on_the_fly && buf_dma != nullptr) {
        nvm_dma_unmap(buf_dma);
    }

    return result;
}

extern "C" ssize_t uGDSRead(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                              off_t file_offset, off_t bufPtr_offset)
{
    return do_io_internal(fh, bufPtr_base, size, file_offset, bufPtr_offset, NVM_IO_READ);
}

extern "C" ssize_t uGDSWrite(uGDSHandle_t fh, const void* bufPtr_base, size_t size,
                               off_t file_offset, off_t bufPtr_offset)
{
    return do_io_internal(fh, const_cast<void*>(bufPtr_base), size, file_offset, bufPtr_offset,
                 NVM_IO_WRITE);
}
