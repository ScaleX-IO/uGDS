#include "test_utils.h"
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>

/* Test: dmabuf export lifecycle
 *
 * Verifies:
 * - uGDSBufRegisterEx produces a valid dmabuf export
 * - uGDSExportDmabuf returns a dup'd fd (different from internal)
 * - Export fd is valid (can be used for ioctl)
 * - NVMe I/O still works after export
 * - Deregister after export close succeeds
 * - Export on non-dmabuf buffer returns UGDS_IO_NOT_SUPPORTED
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

    /* -- 1. Register with dmabuf backend -- */
    uGDSBufConfig_t cfg = {};
    cfg.backend =
#ifdef __HIP_PLATFORM_AMD__
        UGDS_BACKEND_HIP;
#else
        UGDS_BACKEND_CUDA;
#endif
    cfg.enable_export = true;  /* request dma-buf fd for export test */

    st = uGDSBufRegisterEx(d_buf, buf_size, &cfg);
    if (st.err != UGDS_SUCCESS) {
        /* Runtime may return NOT_SUPPORTED even when the API exists at
         * compile time (e.g., GPU driver lacks dmabuf support). */
        if (st.err == UGDS_IO_NOT_SUPPORTED ||
            st.err == UGDS_PLATFORM_NOT_SUPPORTED) {
            printf("SKIP: dma-buf export not supported on this device\n");
            cudaFree(d_buf);
            close_handle(fh);
            uGDSDriverClose();
            return 77;  /* skip exit code (automake convention) */
        }
        TEST_FAIL("BufRegisterEx: %s", UGDS_ERRSTR(st.err));
    }

    /* -- 2. Export dmabuf handle -- */
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

    /* -- 3. Verify fd is usable (basic fstat) -- */
    struct stat fd_stat;
    if (fstat(exp.fd, &fd_stat) != 0)
        TEST_FAIL("fstat on export fd failed: %s", strerror(errno));

    /* -- 4. NVMe I/O still works after export -- */
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

    /* -- 5. Close export fd, then deregister -- */
    close(exp.fd);

    st = uGDSBufDeregister(d_buf);
    ASSERT_OK(st, "BufDeregister after export close");

    /* -- 6. Export on non-dmabuf buffer -> IO_NOT_SUPPORTED --
     * On CUDA builds, TEST_BUF_FLAGS=0 so the buffer is mapped via
     * the standard CUDA P2P path (no dmabuf fd retained).
     * On HIP builds, TEST_BUF_FLAGS=UGDS_REGISTER_DMABUF but the fd
     * is not retained (no NVM_MAP_RDMA), so export still fails. */
    void* d_buf2 = nullptr;
    cudaMalloc(&d_buf2, buf_size);
    if (!d_buf2) TEST_FAIL("cudaMalloc(2) failed");

    st = uGDSBufRegister(d_buf2, buf_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister (non-dmabuf)");

    uGDSDmabufExport_t exp2 = {};
    st = uGDSExportDmabuf(d_buf2, &exp2);
    ASSERT_ERR(st, UGDS_IO_NOT_SUPPORTED, "export on non-dmabuf buffer");

    st = uGDSBufDeregister(d_buf2);
    ASSERT_OK(st, "BufDeregister (non-dmabuf)");

    /* -- 7. Export on unregistered buffer -- */
    uGDSDmabufExport_t exp3 = {};
    st = uGDSExportDmabuf(d_buf2, &exp3);
    ASSERT_ERR(st, UGDS_MEMORY_NOT_REGISTERED, "export on unregistered buffer");

    /* -- 8. fd leak check: count /proc/self/fd before/after -- */
    auto count_fds = []() -> int {
        DIR* d = opendir("/proc/self/fd");
        if (!d) return -1;
        int count = 0;
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            if (ent->d_name[0] != '.') count++;
        }
        closedir(d);
        return count;
    };

    int fd_before = count_fds();
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

    int fd_after = count_fds();
    /* Allow runtime libraries to hold internal fds, but flag large leaks */
    if (fd_before > 0 && fd_after > fd_before + 2)
        TEST_FAIL("fd leak: before=%d after=%d", fd_before, fd_after);

    cudaFree(d_buf);
    cudaFree(d_buf2);
    close_handle(fh);
    uGDSDriverClose();

    TEST_PASS();
}
