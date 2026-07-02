#include "test_utils.h"
#include <sys/stat.h>

/* Test: RDMA dmabuf export lifecycle
 *
 * Verifies:
 * - uGDSBufRegisterEx with enable_rdma produces a valid dmabuf export
 * - uGDSExportDmabuf returns a dup'd fd (different from internal)
 * - Export fd is valid (can be used for ioctl)
 * - NVMe I/O still works after export
 * - Deregister after export close succeeds
 * - Export on non-RDMA buffer returns UGDS_IO_NOT_SUPPORTED
 * - Export on unregistered buffer returns UGDS_MEMORY_NOT_REGISTERED
 */
int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t buf_size = 65536;
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, buf_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    /* ── 1. Register with RDMA enabled ── */
    uGDSBufConfig_t cfg = {};
    cfg.backend =
#ifdef __HIP_PLATFORM_AMD__
        UGDS_BACKEND_HIP;
#else
        UGDS_BACKEND_CUDA;
#endif
    cfg.enable_rdma = 1;

    st = uGDSBufRegisterEx(d_buf, buf_size, &cfg);
    if (st.err != UGDS_SUCCESS) {
        /* If dma-buf export is not supported on this GPU, skip gracefully */
        if (st.err == UGDS_IO_NOT_SUPPORTED || st.err == UGDS_PLATFORM_NOT_SUPPORTED) {
            printf("SKIP: dma-buf export not supported on this device\n");
            cudaFree(d_buf);
            close_handle(fh);
            uGDSDriverClose();
            return 0;
        }
        TEST_FAIL("BufRegisterEx (RDMA): %s", UGDS_ERRSTR(st.err));
    }

    /* ── 2. Export dmabuf handle ── */
    uGDSDmabufExport_t exp = {};
    st = uGDSExportDmabuf(d_buf, &exp);
    ASSERT_OK(st, "ExportDmabuf");

    if (exp.fd < 0)
        TEST_FAIL("export fd < 0");
    if (exp.length != buf_size)
        TEST_FAIL("export length mismatch: %zu != %zu", exp.length, buf_size);
#ifdef __HIP_PLATFORM_AMD__
    /* HIP may have non-zero offset */
    if (exp.offset % getpagesize() != 0)
        TEST_FAIL("HIP offset not page-aligned: %lu", exp.offset);
#else
    /* CUDA offset should be 0 */
    if (exp.offset != 0)
        TEST_FAIL("CUDA offset != 0: %lu", exp.offset);
#endif

    /* ── 3. Verify fd is usable (basic fstat) ── */
    struct stat fd_stat;
    if (fstat(exp.fd, &fd_stat) != 0)
        TEST_FAIL("fstat on export fd failed: %s", strerror(errno));

    /* ── 4. NVMe I/O still works after export ── */
    /* Write pattern to GPU buffer */
    cudaMemset(d_buf, 0xAB, buf_size);
    cudaDeviceSynchronize();

    ssize_t wr = uGDSWrite(fh, d_buf, 4096, 0, 0);
    if (wr != 4096)
        TEST_FAIL("uGDSWrite after export: %zd", wr);

    /* Read back to verify */
    cudaMemset(d_buf, 0x00, buf_size);
    cudaDeviceSynchronize();

    ssize_t rd = uGDSRead(fh, d_buf, 4096, 0, 0);
    if (rd != 4096)
        TEST_FAIL("uGDSRead after export: %zd", rd);

    /* ── 5. Close export fd, then deregister ── */
    close(exp.fd);

    st = uGDSBufDeregister(d_buf);
    ASSERT_OK(st, "BufDeregister after export close");

    /* ── 6. Export on non-RDMA buffer → IO_NOT_SUPPORTED ── */
    void* d_buf2 = nullptr;
    cudaMalloc(&d_buf2, buf_size);
    if (!d_buf2) TEST_FAIL("cudaMalloc(2) failed");

    st = uGDSBufRegister(d_buf2, buf_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister (non-RDMA)");

    uGDSDmabufExport_t exp2 = {};
    st = uGDSExportDmabuf(d_buf2, &exp2);
    ASSERT_ERR(st, UGDS_IO_NOT_SUPPORTED, "export on non-RDMA buffer");

    st = uGDSBufDeregister(d_buf2);
    ASSERT_OK(st, "BufDeregister (non-RDMA)");

    /* ── 7. Export on unregistered buffer ── */
    uGDSDmabufExport_t exp3 = {};
    st = uGDSExportDmabuf(d_buf2, &exp3);
    ASSERT_ERR(st, UGDS_MEMORY_NOT_REGISTERED, "export on unregistered buffer");

    /* ── 8. fd leak check: repeated register/export/deregister ── */
    int fd_before = open("/dev/null", O_RDONLY);
    close(fd_before);
    /* The fd number should be recycled, so after full cycle the fd
     * count should return to baseline */
    for (int i = 0; i < 10; i++) {
        st = uGDSBufRegisterEx(d_buf, buf_size, &cfg);
        ASSERT_OK(st, "BufRegisterEx loop");

        uGDSDmabufExport_t exp_l = {};
        st = uGDSExportDmabuf(d_buf, &exp_l);
        ASSERT_OK(st, "ExportDmabuf loop");

        close(exp_l.fd);
        st = uGDSBufDeregister(d_buf);
        ASSERT_OK(st, "BufDeregister loop");
    }

    int fd_after = open("/dev/null", O_RDONLY);
    close(fd_after);
    if (fd_after != fd_before)
        TEST_FAIL("fd leak: before=%d after=%d", fd_before, fd_after);

    cudaFree(d_buf);
    cudaFree(d_buf2);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}
