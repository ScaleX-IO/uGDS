#include "ugds_internal.h"
#include "libnvm/internal/ioctl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/eventfd.h>
#include <unistd.h>

static void cleanup_qp(nvm_aq_ref aq_ref, IOQueuePair* qp, int dev_fd) {
    if (qp->sq_dma) {
        nvm_admin_sq_delete(aq_ref, &qp->sq, &qp->cq);
        nvm_dma_unmap(qp->sq_dma);
    }
    if (qp->cq_dma) {
        nvm_admin_cq_delete(aq_ref, &qp->cq);
        nvm_dma_unmap(qp->cq_dma);
    }
    if (qp->irq_efd >= 0) {
        struct nvm_ioctl_irq req = {};
        req.vector = qp->irq_vec;
        req.eventfd = qp->irq_efd;
        if (dev_fd >= 0)
            ioctl(dev_fd, NVM_UNREGISTER_INTERRUPT, &req);
        close(qp->irq_efd);
        qp->irq_efd = -1;
    }
    if (qp->prp_dma) nvm_dma_unmap(qp->prp_dma);
    free(qp->sq_buf);
    free(qp->cq_buf);
    free(qp->prp_buf);
}

static void cleanup_batch_qp(nvm_aq_ref aq_ref, IOQueuePairHuge* bqp, int dev_fd) {
    // Null out hugepage-backed bufs so cleanup_qp calls free(nullptr) for those
    if (bqp->sq_huge) bqp->qp.sq_buf = nullptr;
    if (bqp->cq_huge) bqp->qp.cq_buf = nullptr;
    cleanup_qp(aq_ref, &bqp->qp, dev_fd);

    if (bqp->sq_huge)
        hugepage_free(bqp->sq_huge, bqp->sq_huge_size);
    if (bqp->cq_huge)
        hugepage_free(bqp->cq_huge, bqp->cq_huge_size);
}

extern "C" uGDSError_t uGDSHandleRegister(uGDSHandle_t* fh, uGDSDescr_t* descr)
{
    if (fh == nullptr || descr == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (descr->type != UGDS_HANDLE_TYPE_OPAQUE_FD)
        return make_error(UGDS_INVALID_FILE_TYPE);

    auto hs = std::make_unique<HandleState>();
    hs->fd = descr->handle.fd;
    hs->ns_id = 1;
    hs->ctrl = nullptr;
    hs->aq_ref = nullptr;
    hs->aq_dma = nullptr;
    hs->aq_buf = nullptr;

    int status = nvm_ctrl_init(&hs->ctrl, hs->fd);
    if (!nvm_ok(status))
        return make_error(UGDS_INTERNAL_ERROR);

    const size_t page_size = hs->ctrl->page_size;

    if (posix_memalign(&hs->aq_buf, 4096, page_size * 3) != 0) {
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }
    std::memset(hs->aq_buf, 0, page_size * 3);

    status = nvm_dma_map_host(&hs->aq_dma, hs->ctrl, hs->aq_buf, page_size * 3);
    if (!nvm_ok(status)) {
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    status = nvm_aq_create(&hs->aq_ref, hs->ctrl, hs->aq_dma);
    if (!nvm_ok(status)) {
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    status = nvm_admin_ctrl_info(hs->aq_ref, &hs->ctrl_info,
                                 NVM_DMA_OFFSET(hs->aq_dma, 2),
                                 hs->aq_dma->ioaddrs[2]);
    if (!nvm_ok(status)) {
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    status = nvm_admin_ns_info(hs->aq_ref, &hs->ns_info, hs->ns_id,
                               NVM_DMA_OFFSET(hs->aq_dma, 2),
                               hs->aq_dma->ioaddrs[2]);
    if (!nvm_ok(status)) {
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    hs->block_size = hs->ns_info.lba_data_size;
    hs->max_transfer_size = hs->ctrl_info.max_data_size;
    hs->max_transfer_pages = hs->ctrl_info.max_data_pages;

    uint16_t n_cqs = UGDS_DEFAULT_NUM_QPS;
    uint16_t n_sqs = UGDS_DEFAULT_NUM_QPS;
    status = nvm_admin_request_num_queues(hs->aq_ref, &n_cqs, &n_sqs);
    if (!nvm_ok(status)) {
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    uint16_t avail = std::min(n_cqs, n_sqs);
    // Need at least 2 IO QPs: 1 sync + 1 batch
    if (avail < 2) {
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }
    uint16_t total_qps = std::min<uint16_t>(avail, UGDS_DEFAULT_NUM_QPS);
    uint16_t sync_qps = total_qps - 1;
    hs->num_qps = sync_qps;

    uint16_t irq_vectors = 0;
    {
        const char* env = getenv("UGDS_INTERRUPT_MODE");
        bool want = env != nullptr &&
            (strcmp(env, "1") == 0 || strcmp(env, "on") == 0 ||
             strcmp(env, "true") == 0 || strcmp(env, "yes") == 0);
        if (want) {
            uint32_t nv = 0;
            if (ioctl(hs->fd, NVM_GET_NUM_VECTORS, &nv) == 0 && nv > 0) {
                hs->interrupt_mode = true;
                irq_vectors = std::min<uint16_t>(sync_qps,
                                                 static_cast<uint16_t>(nv));
            } else {
                fprintf(stderr, "uGDS: interrupt mode requested but no MSI-X "
                                "vectors available; using poll mode\n");
            }
        }
    }

    // Create sync IO QPs (shallow depth, 4KB pages)
    for (uint16_t i = 0; i < sync_qps; ++i) {
        auto qp = std::make_unique<IOQueuePair>();

        // CQ
        if (posix_memalign(&qp->cq_buf, 4096, page_size) != 0)
            goto fail;
        std::memset(qp->cq_buf, 0, page_size);
        status = nvm_dma_map_host(&qp->cq_dma, hs->ctrl, qp->cq_buf, page_size);
        if (!nvm_ok(status)) goto fail;

        if (hs->interrupt_mode && i < irq_vectors) {
            int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
            if (efd >= 0) {
                struct nvm_ioctl_irq req = {};
                req.vector = i;
                req.eventfd = efd;
                if (ioctl(hs->fd, NVM_REGISTER_INTERRUPT, &req) == 0) {
                    qp->irq_efd = efd;
                    qp->irq_vec = i;
                } else {
                    close(efd);
                }
            }
        }

        status = nvm_admin_cq_create(hs->aq_ref, &qp->cq, i + 1,
                                     qp->cq_dma, 0, UGDS_DEFAULT_QUEUE_DEPTH,
                                     false, qp->irq_vec, qp->irq_efd >= 0);
        if (!nvm_ok(status)) {
            nvm_dma_unmap(qp->cq_dma);
            qp->cq_dma = nullptr;
            goto fail;
        }

        // SQ
        if (posix_memalign(&qp->sq_buf, 4096, page_size) != 0) goto fail;
        std::memset(qp->sq_buf, 0, page_size);
        status = nvm_dma_map_host(&qp->sq_dma, hs->ctrl, qp->sq_buf, page_size);
        if (!nvm_ok(status)) goto fail;
        status = nvm_admin_sq_create(hs->aq_ref, &qp->sq, &qp->cq, i + 1,
                                     qp->sq_dma, 0, UGDS_DEFAULT_QUEUE_DEPTH);
        if (!nvm_ok(status)) {
            nvm_dma_unmap(qp->sq_dma);
            qp->sq_dma = nullptr;
            goto fail;
        }

        // PRP list page
        if (posix_memalign(&qp->prp_buf, 4096, page_size) != 0) goto fail;
        std::memset(qp->prp_buf, 0, page_size);
        status = nvm_dma_map_host(&qp->prp_dma, hs->ctrl, qp->prp_buf, page_size);
        if (!nvm_ok(status)) goto fail;

        hs->qps.push_back(std::move(qp));
        continue;

    fail:
        cleanup_qp(hs->aq_ref, qp.get(), hs->fd);
        for (auto& done : hs->qps)
            cleanup_qp(hs->aq_ref, done.get(), hs->fd);
        hs->qps.clear();
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    }

    // Create batch IO QP (deep depth, hugepage-backed)
    {
        uint16_t batch_qp_id = sync_qps + 1;
        uint16_t batch_depth = std::min<uint16_t>(
            UGDS_BATCH_QUEUE_DEPTH, hs->ctrl->max_qs);
        auto bqp = std::make_unique<IOQueuePairHuge>();
        size_t sq_data_size = batch_depth * sizeof(nvm_cmd_t);
        size_t cq_data_size = batch_depth * sizeof(nvm_cpl_t);

        bool need_hugepage_sq = (sq_data_size > page_size);
        bool need_hugepage_cq = false;

        // SQ allocation: try hugepage, fall back to 4KB page with reduced depth
        if (need_hugepage_sq) {
            bqp->qp.sq_buf = hugepage_alloc(sq_data_size, &bqp->sq_huge_size);
            if (bqp->qp.sq_buf)
                bqp->sq_huge = bqp->qp.sq_buf;
        }
        if (!bqp->qp.sq_buf) {
            if (need_hugepage_sq)
                fprintf(stderr, "uGDS: hugepage alloc failed for batch SQ, "
                        "falling back to 4KB page (depth %u → %zu)\n",
                        batch_depth, page_size / sizeof(nvm_cmd_t));
            size_t alloc = std::max(sq_data_size, page_size);
            if (posix_memalign(&bqp->qp.sq_buf, 4096, alloc) != 0)
                goto batch_fail;
            std::memset(bqp->qp.sq_buf, 0, alloc);
            if (need_hugepage_sq)
                batch_depth = static_cast<uint16_t>(page_size / sizeof(nvm_cmd_t));
            sq_data_size = batch_depth * sizeof(nvm_cmd_t);
            cq_data_size = batch_depth * sizeof(nvm_cpl_t);
        }

        // CQ allocation: try hugepage if SQ got hugepage, else 4KB page
        need_hugepage_cq = (cq_data_size > page_size);
        if (need_hugepage_cq && bqp->sq_huge) {
            bqp->qp.cq_buf = hugepage_alloc(cq_data_size, &bqp->cq_huge_size);
            if (bqp->qp.cq_buf)
                bqp->cq_huge = bqp->qp.cq_buf;
        }
        if (!bqp->qp.cq_buf) {
            if (need_hugepage_cq)
                fprintf(stderr, "uGDS: hugepage alloc failed for batch CQ, "
                        "falling back to 4KB page\n");
            size_t alloc = std::max(cq_data_size, page_size);
            if (posix_memalign(&bqp->qp.cq_buf, 4096, alloc) != 0)
                goto batch_fail;
            std::memset(bqp->qp.cq_buf, 0, alloc);
            if (need_hugepage_cq)
                batch_depth = std::min(batch_depth,
                    static_cast<uint16_t>(page_size / sizeof(nvm_cpl_t)));
        }
        sq_data_size = batch_depth * sizeof(nvm_cmd_t);
        cq_data_size = batch_depth * sizeof(nvm_cpl_t);

        status = nvm_dma_map_host(&bqp->qp.cq_dma, hs->ctrl,
                                   bqp->qp.cq_buf, cq_data_size);
        if (!nvm_ok(status)) goto batch_fail;
        status = nvm_admin_cq_create(hs->aq_ref, &bqp->qp.cq, batch_qp_id,
                                     bqp->qp.cq_dma, 0, batch_depth);
        if (!nvm_ok(status)) {
            nvm_dma_unmap(bqp->qp.cq_dma);
            bqp->qp.cq_dma = nullptr;
            goto batch_fail;
        }

        status = nvm_dma_map_host(&bqp->qp.sq_dma, hs->ctrl,
                                   bqp->qp.sq_buf, sq_data_size);
        if (!nvm_ok(status)) goto batch_fail;
        status = nvm_admin_sq_create(hs->aq_ref, &bqp->qp.sq, &bqp->qp.cq,
                                     batch_qp_id, bqp->qp.sq_dma, 0, batch_depth);
        if (!nvm_ok(status)) {
            nvm_dma_unmap(bqp->qp.sq_dma);
            bqp->qp.sq_dma = nullptr;
            goto batch_fail;
        }

        // PRP list page for batch QP (regular 4KB page is fine)
        if (posix_memalign(&bqp->qp.prp_buf, 4096, page_size) != 0)
            goto batch_fail;
        std::memset(bqp->qp.prp_buf, 0, page_size);
        status = nvm_dma_map_host(&bqp->qp.prp_dma, hs->ctrl,
                                   bqp->qp.prp_buf, page_size);
        if (!nvm_ok(status)) goto batch_fail;

        hs->batch_qp = std::move(bqp);
        hs->batch_queue_depth = batch_depth;
        goto batch_done;

    batch_fail:
        cleanup_batch_qp(hs->aq_ref, bqp.get(), hs->fd);
        for (auto& done : hs->qps)
            cleanup_qp(hs->aq_ref, done.get(), hs->fd);
        hs->qps.clear();
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
    batch_done:;
    }

    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        if (g_driver.default_ctrl == nullptr)
            g_driver.default_ctrl = hs->ctrl;
    }

    *fh = reinterpret_cast<uGDSHandle_t>(hs.release());
    return UGDS_OK;
}

extern "C" void uGDSHandleDeregister(uGDSHandle_t fh)
{
    if (fh == nullptr) return;

    HandleState* hs = reinterpret_cast<HandleState*>(fh);

    if (hs->batch_qp)
        cleanup_batch_qp(hs->aq_ref, hs->batch_qp.get(), hs->fd);
    hs->batch_qp.reset();

    for (auto it = hs->qps.rbegin(); it != hs->qps.rend(); ++it)
        cleanup_qp(hs->aq_ref, it->get(), hs->fd);
    hs->qps.clear();

    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        if (g_driver.default_ctrl == hs->ctrl)
            g_driver.default_ctrl = nullptr;
    }

    if (hs->aq_ref) nvm_aq_destroy(hs->aq_ref);
    if (hs->aq_dma) nvm_dma_unmap(hs->aq_dma);
    free(hs->aq_buf);
    if (hs->ctrl) nvm_ctrl_free(hs->ctrl);

    delete hs;
}
