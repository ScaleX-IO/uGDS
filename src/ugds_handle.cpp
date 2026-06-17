#include "ugds_internal.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>

static void cleanup_qp(nvm_aq_ref aq_ref, IOQueuePair* qp) {
    if (qp->sq_dma) {
        nvm_admin_sq_delete(aq_ref, &qp->sq, &qp->cq);
        nvm_dma_unmap(qp->sq_dma);
    }
    if (qp->cq_dma) {
        nvm_admin_cq_delete(aq_ref, &qp->cq);
        nvm_dma_unmap(qp->cq_dma);
    }
    if (qp->prp_dma) nvm_dma_unmap(qp->prp_dma);
    free(qp->sq_buf);
    free(qp->cq_buf);
    free(qp->prp_buf);
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
    hs->num_qps = std::min<uint16_t>(avail, UGDS_DEFAULT_NUM_QPS);

    for (uint16_t i = 0; i < hs->num_qps; ++i) {
        auto qp = std::make_unique<IOQueuePair>();
        qp->sq_dma = nullptr;
        qp->cq_dma = nullptr;
        qp->prp_dma = nullptr;
        qp->sq_buf = nullptr;
        qp->cq_buf = nullptr;
        qp->prp_buf = nullptr;

        // CQ
        if (posix_memalign(&qp->cq_buf, 4096, page_size) != 0)
            goto fail;
        std::memset(qp->cq_buf, 0, page_size);
        status = nvm_dma_map_host(&qp->cq_dma, hs->ctrl, qp->cq_buf, page_size);
        if (!nvm_ok(status)) goto fail;
        status = nvm_admin_cq_create(hs->aq_ref, &qp->cq, i + 1,
                                     qp->cq_dma, 0, UGDS_DEFAULT_QUEUE_DEPTH);
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
        cleanup_qp(hs->aq_ref, qp.get());
        for (auto& done : hs->qps)
            cleanup_qp(hs->aq_ref, done.get());
        hs->qps.clear();
        nvm_aq_destroy(hs->aq_ref);
        nvm_dma_unmap(hs->aq_dma);
        free(hs->aq_buf);
        nvm_ctrl_free(hs->ctrl);
        return make_error(UGDS_INTERNAL_ERROR);
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

    for (auto it = hs->qps.rbegin(); it != hs->qps.rend(); ++it)
        cleanup_qp(hs->aq_ref, it->get());
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
