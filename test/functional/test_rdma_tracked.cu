#include "test_utils.h"
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>

#ifdef _RDMA
#include <infiniband/verbs.h>

/* Test: Tracked RDMA MR lifecycle
 *
 * Verifies:
 * - uGDSRDMARegister returns a valid MR
 * - uGDSRDMAUnregister succeeds after quiescence
 * - uGDSBufDeregister fails with UGDS_RDMA_MR_STILL_ACTIVE while MR is live
 * - uGDSBufDeregister succeeds after unregister
 *
 * NOTE: This test does NOT post any WRs, so quiescence is trivially satisfied.
 * Requires libibverbs but NOT a real RDMA NIC -- ibv_alloc_pd works on any
 * verbs device (or fails gracefully if none present).
 */
int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    /* Find a verbs device for PD allocation */
    struct ibv_device** dev_list = ibv_get_device_list(NULL);
    if (!dev_list || !dev_list[0]) {
        printf("SKIP: no RDMA device available\n");
        if (dev_list) ibv_free_device_list(dev_list);
        close_handle(fh);
        uGDSDriverClose();
        return 77;
    }

    struct ibv_context* ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!ctx) TEST_FAIL("ibv_open_device failed");

    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) TEST_FAIL("ibv_alloc_pd failed");

    const size_t buf_size = 65536;
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, buf_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    /* -- 1. Register buffer with RDMA -- */
    uGDSBufConfig_t cfg = {};
    cfg.backend =
#ifdef __HIP_PLATFORM_AMD__
        UGDS_BACKEND_HIP;
#else
        UGDS_BACKEND_CUDA;
#endif
    cfg.enable_export = true;

    st = uGDSBufRegisterEx(d_buf, buf_size, &cfg);
    if (st.err != UGDS_SUCCESS) {
        if (st.err == UGDS_IO_NOT_SUPPORTED || st.err == UGDS_PLATFORM_NOT_SUPPORTED) {
            printf("SKIP: dma-buf export not supported\n");
            ibv_dealloc_pd(pd);
            ibv_close_device(ctx);
            cudaFree(d_buf);
            close_handle(fh);
            uGDSDriverClose();
            return 77;
        }
        TEST_FAIL("BufRegisterEx: %s", UGDS_ERRSTR(st.err));
    }

    /* -- 2. Tracked RDMA Register -- */
    uGDSRDMARegion_t region = {};
    st = uGDSRDMARegister(d_buf, pd,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE, &region);
    if (st.err == UGDS_IO_NOT_SUPPORTED) {
        printf("SKIP: verbs provider does not support dmabuf MR\n");
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        uGDSBufDeregister(d_buf);
        cudaFree(d_buf);
        close_handle(fh);
        uGDSDriverClose();
        return 77;
    }
    ASSERT_OK(st, "RDMARegister");

    if (!region.mr) TEST_FAIL("MR is null");
    if (region.iova == 0) TEST_FAIL("iova is 0 (should be buffer ptr)");

    /* -- 3. Deregister must fail while MR is active -- */
    st = uGDSBufDeregister(d_buf);
    ASSERT_ERR(st, UGDS_RDMA_MR_STILL_ACTIVE, "deregister with active MR");

    /* -- 4. Tracked RDMA Unregister -- */
    st = uGDSRDMAUnregister(d_buf, &region);
    ASSERT_OK(st, "RDMAUnregister");

    if (region.mr != nullptr) TEST_FAIL("MR not cleared after unregister");

    /* -- 5. Deregister now succeeds -- */
    st = uGDSBufDeregister(d_buf);
    ASSERT_OK(st, "BufDeregister after unregister");

    /* -- 6. Double unregister fails (region->mr already cleared) -- */
    st = uGDSRDMAUnregister(d_buf, &region);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "double unregister (mr already cleared)");

    /* Cleanup */
    cudaFree(d_buf);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}

#else /* !_RDMA */

int main(void) {
    printf("SKIP: build with UGDS_ENABLE_RDMA=ON for tracked MR tests\n");
    return 77;
}

#endif /* _RDMA */
