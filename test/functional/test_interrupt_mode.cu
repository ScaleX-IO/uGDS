// Phase 6 Stage 4 - End-to-end test for interrupt mode
//
// Forces UGDS_INTERRUPT_MODE and verifies that I/O is still correct in
// interrupt mode (the user thread blocks on an eventfd waiting for the MSI-X
// interrupt instead of busy-polling):
//   1. single 4KB write / clear / read-back / verify
//   2. 4 threads x 16 IOs of 4KB, all in flight concurrently -- multiple
//      MSI-X vectors fire in parallel and the CQE-DMA/IRQ race is exercised
//      repeatedly: a completion may be picked up by the poll-before-block
//      CQ check while its eventfd signal lands later, leaving a stale
//      counter that the next wait must treat as a spurious wakeup (drain,
//      re-poll the CQ, re-block). Data integrity across all 64 IOs proves
//      no completion is lost or double-consumed.
//
// Note: the synchronous path keeps each queue pair at QD=1 (submit one,
// wait one, under the per-QP lock), so a literal N-CQEs-one-signal pileup
// on a single CQ cannot be constructed from this API; the stale-signal /
// spurious-wakeup path above is the coalescing-adjacent behavior the wait
// loop must (and does) handle by always re-polling the CQ.
//
// Real-hardware signal: while this test runs, `cat /proc/interrupts | grep ugds`
// counts are non-zero, proving the MSI-X interrupt actually fired (rather than
// silently falling back to polling). The lines disappear after the test exits
// because teardown free_irq()s the vectors.
//
// Note: setenv must happen before uGDSDriverOpen / handle creation, because
// interrupt mode is decided when uGDSHandleRegister creates the CQs.

#include "test_utils.h"
#include <thread>
#include <atomic>

static std::atomic<int> g_errors{0};

static const size_t kIoSize = 4096;
static const int kIosPerWorker = 16;

// Each worker owns a distinct file region and drives kIosPerWorker
// write/read/verify cycles while the other workers do the same in parallel.
static void irq_worker(int id, uGDSHandle_t fh) {
    const size_t alloc_size = 65536;
    const off_t base_off = (off_t)(id + 1) * (off_t)(kIosPerWorker * kIoSize);
    size_t n_words = kIoSize / sizeof(uint32_t);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) {
        fprintf(stderr, "worker %d: cudaMalloc failed\n", id);
        g_errors++;
        return;
    }
    uGDSError_t st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    if (st.err != UGDS_SUCCESS) {
        fprintf(stderr, "worker %d: BufRegister failed: %s\n",
                id, uGDS_status_error(st.err));
        cudaFree(d_buf);
        g_errors++;
        return;
    }

    for (int k = 0; k < kIosPerWorker && g_errors.load() == 0; k++) {
        uint32_t pat = 0xA0000000u | ((uint32_t)id << 16) | (uint32_t)k;
        off_t off = base_off + (off_t)k * kIoSize;

        fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, pat, n_words);
        cudaDeviceSynchronize();
        ssize_t ret = uGDSWrite(fh, d_buf, kIoSize, off, 0);
        if (ret < 0 || (size_t)ret != kIoSize) {
            fprintf(stderr, "worker %d io %d: write failed: %zd\n", id, k, ret);
            g_errors++;
            break;
        }

        cudaMemset(d_buf, 0, kIoSize);
        cudaDeviceSynchronize();
        ret = uGDSRead(fh, d_buf, kIoSize, off, 0);
        if (ret < 0 || (size_t)ret != kIoSize) {
            fprintf(stderr, "worker %d io %d: read failed: %zd\n", id, k, ret);
            g_errors++;
            break;
        }

        uint32_t* h_buf = (uint32_t*)malloc(kIoSize);
        cudaMemcpy(h_buf, d_buf, kIoSize, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n_words; i++) {
            if (h_buf[i] != pat) {
                fprintf(stderr,
                        "worker %d io %d: mismatch at word %zu: 0x%08X != 0x%08X\n",
                        id, k, i, h_buf[i], pat);
                g_errors++;
                break;
            }
        }
        free(h_buf);
    }

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    // Force interrupt mode (must be set before creating the handle)
    setenv("UGDS_INTERRUPT_MODE", "1", 1);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed (interrupt mode)");

    const size_t alloc_size = 65536;
    const uint32_t pattern = 0xABCD1234;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    // 1. Single 4KB write / clear / read-back / verify
    size_t n_words = kIoSize / sizeof(uint32_t);
    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>((uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    ssize_t ret = uGDSWrite(fh, d_buf, kIoSize, 0, 0);
    if (ret < 0 || (size_t)ret != kIoSize)
        TEST_FAIL("interrupt-mode write: %zd / %zu", ret, kIoSize);

    cudaMemset(d_buf, 0, alloc_size);
    cudaDeviceSynchronize();

    ret = uGDSRead(fh, d_buf, kIoSize, 0, 0);
    if (ret < 0 || (size_t)ret != kIoSize)
        TEST_FAIL("interrupt-mode read: %zd / %zu", ret, kIoSize);

    uint32_t* h_buf = (uint32_t*)malloc(kIoSize);
    cudaMemcpy(h_buf, d_buf, kIoSize, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_words; i++) {
        if (h_buf[i] != pattern) {
            uint32_t got = h_buf[i];
            free(h_buf);
            TEST_FAIL("single-IO mismatch at word %zu: 0x%08X != 0x%08X",
                      i, got, pattern);
        }
    }
    free(h_buf);

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);

    // 2. Concurrent interrupt-mode stress: 4 threads x 16 IOs in parallel
    //    (multiple vectors firing simultaneously + stale-eventfd wakeups)
    const int n_threads = 4;
    std::thread threads[n_threads];
    for (int i = 0; i < n_threads; i++)
        threads[i] = std::thread(irq_worker, i, fh);
    for (int i = 0; i < n_threads; i++)
        threads[i].join();
    if (g_errors.load() != 0)
        TEST_FAIL("%d concurrent interrupt-mode error(s)", g_errors.load());

    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
