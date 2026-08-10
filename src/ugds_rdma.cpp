#include "ugds_internal.h"

#ifdef _RDMA
#include <infiniband/verbs.h>
#include <unistd.h>
#include <errno.h>
#include <algorithm>

extern "C" uGDSError_t uGDSRDMARegister(const void* bufPtr_base,
                              void* pd,
                              int access_flags,
                              uGDSRDMARegion_t* region)
{
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (!bufPtr_base || !pd || !region)
        return make_error(UGDS_INVALID_VALUE);

    struct ibv_pd* ibv_pd = static_cast<struct ibv_pd*>(pd);

    /* Use a local temp for output -- only overwrite region on success */
    uGDSRDMARegion_t result = {};

    /* Generate unique token + insert PENDING under lock */
    uint64_t my_token = g_driver.rdma_token_counter.fetch_add(1);
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);

        // Verify buffer is registered
        if (g_driver.buf_registry.find(bufPtr_base) == g_driver.buf_registry.end())
            return make_error(UGDS_MEMORY_NOT_REGISTERED);

        DriverState::RDMARecord pending = {
            bufPtr_base, NULL, -1, 0, 0, 0,
            DriverState::RDMA_REC_PENDING, my_token
        };
        g_driver.rdma_records[bufPtr_base].push_back(pending);
    }

    /* Export dmabuf (outside lock) */
    uGDSDmabufExport_t exp;
    uGDSError_t st = uGDSExportDmabuf(bufPtr_base, &exp);
    if (st.err != UGDS_SUCCESS) goto rollback_pending;

    /* Register MR (outside lock) */
    {
        struct ibv_mr* mr = ibv_reg_dmabuf_mr(ibv_pd, exp.offset, exp.length,
            (uint64_t)bufPtr_base, exp.fd, access_flags);
        if (!mr) {
            /* Save errno before any cleanup */
            int mr_errno = errno;
            close(exp.fd);
            /* Map errno to meaningful uGDS errors */
            if (mr_errno == EOPNOTSUPP || mr_errno == ENOTSUP)
                st = make_error(UGDS_IO_NOT_SUPPORTED);
            else
                st = make_error(UGDS_INTERNAL_ERROR);
            goto rollback_pending;
        }

        /* Complete MY pending record */
        {
            std::lock_guard<std::mutex> guard(g_driver.lock);
            auto map_it = g_driver.rdma_records.find(bufPtr_base);

            // Fail-closed: record vanished during registration
            if (map_it == g_driver.rdma_records.end()) {
                ibv_dereg_mr(mr);
                close(exp.fd);
                return make_error(UGDS_INTERNAL_ERROR);
            }

            auto& vec = map_it->second;
            bool found = false;
            for (auto& r : vec) {
                if (r.token == my_token && r.state == DriverState::RDMA_REC_PENDING) {
                    r.mr     = mr;
                    r.dup_fd = exp.fd;
                    r.iova   = 0;
                    r.offset = exp.offset;
                    r.length = exp.length;
                    r.state  = DriverState::RDMA_REC_ACTIVE;
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Fail-closed: our record vanished, clean up and
                // erase any stale token to avoid blocking deregister
                ibv_dereg_mr(mr);
                close(exp.fd);
                vec.erase(std::remove_if(vec.begin(), vec.end(),
                    [my_token](const DriverState::RDMARecord& r) {
                        return r.token == my_token;
                    }), vec.end());
                return make_error(UGDS_INTERNAL_ERROR);
            }
        }

        result.mr   = mr;
        result.lkey = mr->lkey;
        result.rkey = mr->rkey;
        result.iova = (uint64_t)bufPtr_base;
        result.token = my_token;
    }
    /* Only overwrite caller's region on success */
    *region = result;
    return UGDS_OK;

rollback_pending:
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto map_it = g_driver.rdma_records.find(bufPtr_base);
        if (map_it != g_driver.rdma_records.end()) {
            auto& vec = map_it->second;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [my_token](const DriverState::RDMARecord& r) {
                    return r.token == my_token;
                }), vec.end());
        }
    }
    return (st.err != UGDS_SUCCESS) ? st : make_error(UGDS_INTERNAL_ERROR);
}

extern "C" uGDSError_t uGDSRDMAUnregister(const void* bufPtr_base,
                                uGDSRDMARegion_t* region)
{
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (!bufPtr_base || !region || !region->mr)
        return make_error(UGDS_INVALID_VALUE);

    struct ibv_mr* target_mr = (struct ibv_mr*)region->mr;
    uint64_t region_token = region->token;
    int saved_dup_fd = -1;

    /* Find + claim under lock */
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);

        auto map_it = g_driver.rdma_records.find(bufPtr_base);
        if (map_it == g_driver.rdma_records.end())
            return make_error(UGDS_MEMORY_NOT_REGISTERED);

        auto& vec = map_it->second;
        auto rec_it = std::find_if(vec.begin(), vec.end(),
            [&](const DriverState::RDMARecord& r) {
                return r.mr == target_mr && r.token == region_token;
            });

        if (rec_it == vec.end())
            return make_error(UGDS_HANDLE_NOT_REGISTERED);

        // Check state
        if (rec_it->state == DriverState::RDMA_REC_DEREGISTERING)
            return make_error(UGDS_BUSY);
        if (rec_it->state == DriverState::RDMA_REC_PENDING)
            return make_error(UGDS_BUSY);

        // Mark DEREGISTERING
        rec_it->state = DriverState::RDMA_REC_DEREGISTERING;
        saved_dup_fd = rec_it->dup_fd;
    }

    /* ibv_dereg_mr outside lock */
    int result = ibv_dereg_mr(target_mr);

    if (result != 0) {
        // Failure: restore ACTIVE
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto map_it = g_driver.rdma_records.find(bufPtr_base);
        if (map_it != g_driver.rdma_records.end()) {
            for (auto& r : map_it->second) {
                if (r.mr == target_mr && r.token == region_token &&
                    r.state == DriverState::RDMA_REC_DEREGISTERING) {
                    r.state = DriverState::RDMA_REC_ACTIVE;
                    break;
                }
            }
        }
        return make_error(UGDS_INTERNAL_ERROR);
    }

    /* Remove record + close fd */
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto map_it = g_driver.rdma_records.find(bufPtr_base);
        if (map_it != g_driver.rdma_records.end()) {
            auto& vec = map_it->second;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [&](const DriverState::RDMARecord& r) {
                    return r.mr == target_mr && r.token == region_token &&
                           r.state == DriverState::RDMA_REC_DEREGISTERING;
                }), vec.end());
        }
    }

    if (saved_dup_fd >= 0)
        close(saved_dup_fd);

    region->mr   = NULL;
    region->lkey = 0;
    region->rkey = 0;
    region->iova = 0;
    region->token = 0;
    return UGDS_OK;
}

#else /* !_RDMA -- stubs for non-RDMA builds */

extern "C" uGDSError_t uGDSRDMARegister(const void* bufPtr_base,
                              void* pd,
                              int access_flags,
                              uGDSRDMARegion_t* region)
{
    (void)bufPtr_base; (void)pd; (void)access_flags; (void)region;
    return make_error(UGDS_IO_NOT_SUPPORTED);
}

extern "C" uGDSError_t uGDSRDMAUnregister(const void* bufPtr_base,
                                uGDSRDMARegion_t* region)
{
    (void)bufPtr_base; (void)region;
    return make_error(UGDS_IO_NOT_SUPPORTED);
}

#endif /* _RDMA */
