#include "test_utils.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool is_uint(const std::string& value)
{
    if (value.empty()) return false;
    for (char c : value)
        if (c < '0' || c > '9') return false;
    return true;
}

/* Sum counters for every IRQ currently owned by ugds_drv.  This test runs no
 * synchronous I/O after taking a snapshot, so the delta belongs to batch CQ. */
static bool read_ugds_irq_total(uint64_t* total)
{
    std::ifstream input("/proc/interrupts");
    if (!input) return false;

    bool found = false;
    *total = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("ugds_drv") == std::string::npos) continue;
        std::istringstream stream(line.substr(line.find(':') + 1));
        std::string token;
        while (stream >> token) {
            if (!is_uint(token)) break;
            *total += strtoull(token.c_str(), nullptr, 10);
        }
        found = true;
    }
    return found;
}

static bool verify_events(const uGDSIOEvents_t* events, unsigned nr,
                          unsigned expected, size_t io_size)
{
    std::vector<bool> seen(expected, false);
    if (nr != expected) return false;

    for (unsigned i = 0; i < nr; ++i) {
        uintptr_t cookie = reinterpret_cast<uintptr_t>(events[i].cookie);
        if (events[i].status != UGDS_BATCH_COMPLETE ||
            events[i].ret != static_cast<ssize_t>(io_size) ||
            cookie == 0 || cookie > expected || seen[cookie - 1])
            return false;
        seen[cookie - 1] = true;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (!parse_args(argc, argv)) return 1;
    if (cudaSetDevice(g_gpu_id) != cudaSuccess)
        TEST_FAIL("cudaSetDevice failed");

    /* Completion mode is selected while the handle creates its CQs. */
    setenv("UGDS_INTERRUPT_MODE", "1", 1);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");
    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed (interrupt mode)");

    constexpr unsigned kBatch = 128;
    constexpr size_t kIoSize = 4096;
    constexpr off_t kFileBase = 256 * 1024 * 1024;
    const size_t total_bytes = kBatch * kIoSize;
    const size_t words_per_io = kIoSize / sizeof(uint32_t);

    void* d_buf = nullptr;
    if (cudaMalloc(&d_buf, total_bytes) != cudaSuccess)
        TEST_FAIL("cudaMalloc failed");
    st = uGDSBufRegister(d_buf, total_bytes, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, kBatch);
    ASSERT_OK(st, "BatchIOSetUp");

    std::vector<uGDSIOParams_t> params(kBatch);
    std::vector<uGDSIOEvents_t> events(kBatch);
    for (unsigned i = 0; i < kBatch; ++i) {
        uint32_t pattern = 0xB1700000u | i;
        fill_pattern_u32<<<(words_per_io + 255) / 256, 256>>>(
            reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(d_buf) + i * kIoSize),
            pattern, words_per_io);

        params[i] = {};
        params[i].devPtr_base = d_buf;
        params[i].devPtr_offset = static_cast<off_t>(i * kIoSize);
        params[i].file_offset = kFileBase + static_cast<off_t>(i * kIoSize);
        params[i].size = kIoSize;
        params[i].opcode = UGDS_WRITE;
        params[i].cookie = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        TEST_FAIL("pattern initialization failed");

    uint64_t irq_before = 0;
    if (!read_ugds_irq_total(&irq_before))
        TEST_FAIL("no ugds_drv IRQs found; interrupt mode may have fallen back");

    st = uGDSBatchIOSubmit(batch, kBatch, params.data(), 0);
    ASSERT_OK(st, "batch interrupt write submit");
    unsigned nr = kBatch;
    st = uGDSBatchIOGetStatus(batch, kBatch, &nr, events.data(), nullptr);
    ASSERT_OK(st, "batch interrupt write status");
    if (!verify_events(events.data(), nr, kBatch, kIoSize))
        TEST_FAIL("invalid write completion events");

    uint64_t irq_after = 0;
    if (!read_ugds_irq_total(&irq_after) || irq_after <= irq_before)
        TEST_FAIL("batch write completed without a visible hardware IRQ");
    uint64_t write_irqs = irq_after - irq_before;
    if (write_irqs >= kBatch)
        TEST_FAIL("batch write was not coalesced: %llu IRQs for %u commands",
                  static_cast<unsigned long long>(write_irqs), kBatch);

    if (cudaMemset(d_buf, 0, total_bytes) != cudaSuccess)
        TEST_FAIL("cudaMemset failed");
    if (cudaDeviceSynchronize() != cudaSuccess)
        TEST_FAIL("cudaMemset synchronization failed");
    for (auto& param : params) param.opcode = UGDS_READ;

    irq_before = irq_after;
    st = uGDSBatchIOSubmit(batch, kBatch, params.data(), 0);
    ASSERT_OK(st, "batch interrupt read submit");
    nr = kBatch;
    st = uGDSBatchIOGetStatus(batch, kBatch, &nr, events.data(), nullptr);
    ASSERT_OK(st, "batch interrupt read status");
    if (!verify_events(events.data(), nr, kBatch, kIoSize))
        TEST_FAIL("invalid read completion events");
    if (!read_ugds_irq_total(&irq_after) || irq_after <= irq_before)
        TEST_FAIL("batch read completed without a visible hardware IRQ");
    uint64_t read_irqs = irq_after - irq_before;
    if (read_irqs >= kBatch)
        TEST_FAIL("batch read was not coalesced: %llu IRQs for %u commands",
                  static_cast<unsigned long long>(read_irqs), kBatch);

    std::vector<uint32_t> host(total_bytes / sizeof(uint32_t));
    if (cudaMemcpy(host.data(), d_buf, total_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        TEST_FAIL("cudaMemcpy failed");
    for (unsigned i = 0; i < kBatch; ++i) {
        uint32_t expected = 0xB1700000u | i;
        for (size_t word = 0; word < words_per_io; ++word) {
            uint32_t actual = host[i * words_per_io + word];
            if (actual != expected)
                TEST_FAIL("data mismatch at IO %u word %zu: 0x%08x != 0x%08x",
                          i, word, actual, expected);
        }
    }

    /* One command cannot reach the 16-CQE threshold.  It must complete via
     * the controller's 5 ms aggregation timer instead of hanging in ppoll. */
    if (cudaMemset(d_buf, 0, kIoSize) != cudaSuccess)
        TEST_FAIL("single-I/O cudaMemset failed");
    if (cudaDeviceSynchronize() != cudaSuccess)
        TEST_FAIL("single-I/O cudaMemset synchronization failed");
    params[0].opcode = UGDS_READ;
    params[0].cookie = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
    st = uGDSBatchIOSubmit(batch, 1, params.data(), 0);
    ASSERT_OK(st, "timer fallback submit");
    nr = 1;
    struct timespec timeout = {1, 0};
    st = uGDSBatchIOGetStatus(batch, 1, &nr, events.data(), &timeout);
    ASSERT_OK(st, "timer fallback status");
    if (!verify_events(events.data(), nr, 1, kIoSize))
        TEST_FAIL("timer fallback did not return one completion");

    printf("  coalescing: write=%llu IRQs, read=%llu IRQs for %u commands\n",
           static_cast<unsigned long long>(write_irqs),
           static_cast<unsigned long long>(read_irqs), kBatch);

    uGDSBatchIODestroy(batch);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
