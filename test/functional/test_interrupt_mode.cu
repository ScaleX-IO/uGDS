// Phase 6 阶段 4 — 中断模式端到端测试
//
// 强制开启 UGDS_INTERRUPT_MODE，验证在中断模式下（用户线程阻塞在 eventfd 上
// 等待 MSI-X 中断，而非忙轮询）I/O 仍然正确：
//   1. 单次 4KB 写 / 清零 / 读回 / 校验
//   2. 8 个连续 4KB I/O 突发 —— 检验完成合并（多个完成可能只触发一次 eventfd
//      信号，wait_for_completion 必须靠每次先重新轮询 CQ 把它们全部取到）
//
// 真机验证信号：跑此测试后 `cat /proc/interrupts | grep ugds` 计数应非零，
// 证明 MSI-X 中断确实触发（而非静默回退到轮询）。
//
// 注意：必须在 uGDSDriverOpen / 建 handle 之前 setenv，因为中断模式在
// uGDSHandleRegister 建 CQ 时确定。

#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    // 强制中断模式（必须在建 handle 前设置）
    setenv("UGDS_INTERRUPT_MODE", "1", 1);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed (interrupt mode)");

    const size_t alloc_size = 65536;
    const size_t io_size = 4096;
    const uint32_t pattern = 0xABCD1234;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    // ── 1. 单次 4KB 写/清/读/校验 ──
    size_t n_words = io_size / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    ssize_t ret = uGDSWrite(fh, d_buf, io_size, 0, 0);
    if (ret < 0 || (size_t)ret != io_size)
        TEST_FAIL("interrupt-mode write: %zd / %zu", ret, io_size);

    cudaMemset(d_buf, 0, alloc_size);
    cudaDeviceSynchronize();

    ret = uGDSRead(fh, d_buf, io_size, 0, 0);
    if (ret < 0 || (size_t)ret != io_size)
        TEST_FAIL("interrupt-mode read: %zd / %zu", ret, io_size);

    uint32_t* h_buf = (uint32_t*)malloc(io_size);
    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != pattern) {
            free(h_buf);
            TEST_FAIL("single-IO mismatch at word %zu: 0x%08X != 0x%08X",
                      i, h_buf[i], pattern);
        }
    }
    free(h_buf);

    // ── 2. 8×4KB 突发（检验完成合并）──
    // 写 8 个不同 pattern 到 8 个连续 4KB 块，再读回逐块校验。
    const int N = 8;
    for (int b = 0; b < N; b++) {
        uint32_t pat = 0x1000 + b;
        fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, pat, n_words);
        cudaDeviceSynchronize();
        ret = uGDSWrite(fh, d_buf, io_size, (off_t)b * io_size, 0);
        if (ret < 0 || (size_t)ret != io_size)
            TEST_FAIL("burst write block %d: %zd", b, ret);
    }
    for (int b = 0; b < N; b++) {
        uint32_t pat = 0x1000 + b;
        cudaMemset(d_buf, 0, alloc_size);
        cudaDeviceSynchronize();
        ret = uGDSRead(fh, d_buf, io_size, (off_t)b * io_size, 0);
        if (ret < 0 || (size_t)ret != io_size)
            TEST_FAIL("burst read block %d: %zd", b, ret);
        uint32_t* hb = (uint32_t*)malloc(io_size);
        cudaMemcpy(hb, d_buf, io_size, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n_words; i++) {
            if (hb[i] != pat) {
                free(hb);
                TEST_FAIL("burst block %d mismatch at word %zu: 0x%08X != 0x%08X",
                          b, i, hb[i], pat);
            }
        }
        free(hb);
    }

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
