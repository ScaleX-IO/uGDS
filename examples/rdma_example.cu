/* GPUDirect RDMA Example -- Producer/Consumer Pattern
 *
 * Build: Requires UGDS_ENABLE_RDMA=ON and libibverbs.
 */

#include <ugds.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#ifdef UGDS_ENABLE_RDMA
#include <infiniband/verbs.h>

int main(int argc, char** argv)
{
    const char* dev_path = (argc > 1) ? argv[1] : "/dev/ugds_drv0";
    const char* rdma_dev = (argc > 2) ? argv[2] : "mlx5_0";
    const size_t buf_size = 1 << 20;

    int rc = 1;
    uGDSHandle_t fh = NULL;
    int nvme_fd = -1;
    void* d_buf = NULL;
    int dmabuf_fd = -1;
    struct ibv_mr* mr = NULL;
    struct ibv_pd* pd = NULL;
    struct ibv_context* ctx = NULL;
    struct ibv_device** dev_list = NULL;
    bool buf_registered = false;

    printf("uGDS GPUDirect RDMA Example\n");

    /* Use do-while(0) to allow break-based cleanup without goto */
    do {
        /* -- Initialize uGDS -- */
        uGDSError_t st = uGDSDriverOpen();
        if (st.err != UGDS_SUCCESS) {
            fprintf(stderr, "uGDSDriverOpen failed: %s\n", UGDS_ERRSTR(st.err));
            break;
        }

        /* -- Open NVMe handle -- */
        nvme_fd = open(dev_path, O_RDWR);
        if (nvme_fd < 0) { perror("open NVMe"); break; }

        uGDSDescr_t descr;
        memset(&descr, 0, sizeof(descr));
        descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = nvme_fd;
        st = uGDSHandleRegister(&fh, &descr);
        if (st.err != UGDS_SUCCESS) {
            fprintf(stderr, "uGDSHandleRegister failed: %s\n", UGDS_ERRSTR(st.err));
            break;
        }

        /* -- Allocate GPU memory -- */
        if (cudaMalloc(&d_buf, buf_size) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed\n");
            break;
        }

        /* -- Register buffer with RDMA enabled -- */
        uGDSBufConfig_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.backend = UGDS_BACKEND_CUDA;
        cfg.enable_export = true;
        st = uGDSBufRegisterEx(d_buf, buf_size, &cfg);
        if (st.err != UGDS_SUCCESS) {
            fprintf(stderr, "uGDSBufRegisterEx failed: %s\n", UGDS_ERRSTR(st.err));
            break;
        }
        buf_registered = true;

        /* -- Export dma-buf handle -- */
        uGDSDmabufExport_t exp;
        memset(&exp, 0, sizeof(exp));
        st = uGDSExportDmabuf(d_buf, &exp);
        if (st.err != UGDS_SUCCESS) {
            fprintf(stderr, "uGDSExportDmabuf failed: %s\n", UGDS_ERRSTR(st.err));
            break;
        }
        dmabuf_fd = exp.fd;
        printf("  Export: fd=%d, offset=%lu, length=%zu\n",
               exp.fd, (unsigned long)exp.offset, exp.length);

        /* -- Open RDMA device -- */
        dev_list = ibv_get_device_list(NULL);
        if (!dev_list) { fprintf(stderr, "ibv_get_device_list failed\n"); break; }
        for (int i = 0; dev_list[i]; i++) {
            if (strcmp(ibv_get_device_name(dev_list[i]), rdma_dev) == 0) {
                ctx = ibv_open_device(dev_list[i]);
                break;
            }
        }
        if (!ctx) { fprintf(stderr, "RDMA device %s not found\n", rdma_dev); break; }

        pd = ibv_alloc_pd(ctx);
        if (!pd) { perror("ibv_alloc_pd"); break; }

        mr = ibv_reg_dmabuf_mr(pd, exp.offset, exp.length, (uint64_t)d_buf,
            dmabuf_fd,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
            IBV_ACCESS_REMOTE_WRITE);
        if (!mr) { fprintf(stderr, "ibv_reg_dmabuf_mr failed\n"); break; }
        printf("  MR: lkey=%u, rkey=%u\n", mr->lkey, mr->rkey);

        /* -- Producer: NVMe Read -> GPU VRAM -- */
        printf("Phase 1: NVMe Read (producer)...\n");
        ssize_t bytes = uGDSRead(fh, d_buf, 4096, 0, 0);
        if (bytes != 4096) {
            fprintf(stderr, "uGDSRead failed: %zd\n", bytes);
            break;
        }
        printf("  NVMe Read completed: %zd bytes\n", bytes);

        /* -- Consumer: RDMA Send (safe after NVMe completion) -- */
        printf("Phase 2: RDMA Send (consumer) - safe after NVMe completion\n");
        printf("  (RDMA send omitted - requires QP setup)\n");

        /* -- Success -- */
        printf("\nSuccess.\n");
        rc = 0;
    } while (0);

    /* Single cleanup path */
    printf("Phase 3: Teardown\n");
    if (mr) {
        int dereg_err = ibv_dereg_mr(mr);
        if (dereg_err != 0) {
            fprintf(stderr, "Error: ibv_dereg_mr failed: %s\n", strerror(dereg_err));
            fprintf(stderr, "  Aborting teardown -- MR may still reference GPU memory\n");
            return 1;
        }
        printf("  MR deregistered\n");
    }
    if (dmabuf_fd >= 0) { close(dmabuf_fd); printf("  Export fd closed\n"); }
    if (buf_registered) {
        uGDSError_t dst = uGDSBufDeregister(d_buf);
        if (dst.err != UGDS_SUCCESS) {
            fprintf(stderr, "Error: uGDSBufDeregister failed: %s\n", UGDS_ERRSTR(dst.err));
            fprintf(stderr, "  Aborting teardown -- buffer may still be in use\n");
            return 1;
        }
        printf("  uGDS buffer deregistered\n");
    }
    if (d_buf) { cudaFree(d_buf); printf("  GPU memory freed\n"); }
    if (fh) uGDSHandleDeregister(fh);
    if (nvme_fd >= 0) close(nvme_fd);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    if (dev_list) ibv_free_device_list(dev_list);
    uGDSDriverClose();

    if (rc == 0) printf("\nDone. All resources cleaned up.\n");
    return rc;
}

#else /* !UGDS_ENABLE_RDMA */

int main(void)
{
    printf("This example requires UGDS_ENABLE_RDMA=ON at build time.\n");
    printf("Rebuild with: cmake .. -DUGDS_ENABLE_RDMA=ON\n");
    return 0;
}

#endif /* UGDS_ENABLE_RDMA */
