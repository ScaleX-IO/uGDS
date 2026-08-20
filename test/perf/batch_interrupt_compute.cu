#include <ugds.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
    std::string device = "/dev/ugds_drv0";
    int gpu = 0;
    int cpu = 0;
    size_t io_size = 128 * 1024;
    unsigned batch_size = 128;
    double seconds = 3.0;
};

struct Result {
    double gib_s = 0;
    double io_cpu_pct = 0;
    double compute_available_pct = 0;
    double irq_per_batch = 0;
};

void check_ugds(const char* what, uGDSError_t status)
{
    if (status.err != UGDS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": " +
                                 uGDS_status_error(status.err));
}

void check_cuda(const char* what, cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
}

double thread_cpu_seconds()
{
    struct timespec time = {};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) != 0)
        throw std::runtime_error("clock_gettime failed");
    return time.tv_sec + time.tv_nsec * 1e-9;
}

void pin_current_thread(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int status = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (status != 0)
        throw std::runtime_error("cannot pin thread to CPU " +
                                 std::to_string(cpu));
}

bool is_uint(const std::string& value)
{
    if (value.empty()) return false;
    for (char c : value)
        if (c < '0' || c > '9') return false;
    return true;
}

uint64_t irq_total()
{
    std::ifstream input("/proc/interrupts");
    std::string line;
    uint64_t total = 0;
    while (std::getline(input, line)) {
        if (line.find("ugds_drv") == std::string::npos) continue;
        std::istringstream stream(line.substr(line.find(':') + 1));
        std::string token;
        while (stream >> token) {
            if (!is_uint(token)) break;
            total += strtoull(token.c_str(), nullptr, 10);
        }
    }
    return total;
}

class ComputeWorker {
public:
    explicit ComputeWorker(int cpu) : cpu_(cpu) {}

    void start()
    {
        stop_.store(false, std::memory_order_relaxed);
        ops_.store(0, std::memory_order_relaxed);
        thread_ = std::thread([this] {
            pin_current_thread(cpu_);
            uint64_t value = 0x9e3779b97f4a7c15ULL;
            uint64_t ops = 0;
            while (!stop_.load(std::memory_order_relaxed)) {
                for (unsigned i = 0; i < 1024; ++i) {
                    value ^= value << 13;
                    value ^= value >> 7;
                    value ^= value << 17;
                }
                ops += 1024;
                ops_.store(ops, std::memory_order_relaxed);
                asm volatile("" : "+r"(value));
            }
        });
    }

    void stop()
    {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) thread_.join();
    }

    uint64_t ops() const { return ops_.load(std::memory_order_relaxed); }
    ~ComputeWorker() { stop(); }

private:
    int cpu_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> ops_{0};
    std::thread thread_;
};

double compute_baseline(int cpu)
{
    ComputeWorker worker(cpu);
    worker.start();
    auto begin = Clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    uint64_t ops = worker.ops();
    worker.stop();
    double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
    return ops / elapsed;
}

struct Context {
    int fd = -1;
    bool driver_open = false;
    bool buffer_registered = false;
    uGDSHandle_t handle = nullptr;
    uGDSBatchHandle_t batch = nullptr;
    void* gpu_buffer = nullptr;
    size_t buffer_bytes = 0;
    uint64_t capacity = 0;

    ~Context()
    {
        if (batch) uGDSBatchIODestroy(batch);
        if (buffer_registered) uGDSBufDeregister(gpu_buffer);
        if (gpu_buffer) cudaFree(gpu_buffer);
        if (handle) uGDSHandleDeregister(handle);
        if (fd >= 0) close(fd);
        if (driver_open) uGDSDriverClose();
    }

    void open(const Config& cfg, bool interrupt)
    {
        setenv("UGDS_INTERRUPT_MODE", interrupt ? "1" : "0", 1);
        check_ugds("uGDSDriverOpen", uGDSDriverOpen());
        driver_open = true;
        fd = ::open(cfg.device.c_str(), O_RDWR);
        if (fd < 0)
            throw std::runtime_error("cannot open " + cfg.device);

        uGDSDescr_t descriptor = {};
        descriptor.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
        descriptor.handle.fd = fd;
        check_ugds("uGDSHandleRegister",
                   uGDSHandleRegister(&handle, &descriptor));
        check_ugds("uGDSGetDeviceCapacity",
                   uGDSGetDeviceCapacity(handle, &capacity));

        buffer_bytes = cfg.io_size * cfg.batch_size;
        check_cuda("cudaMalloc", cudaMalloc(&gpu_buffer, buffer_bytes));
        check_ugds("uGDSBufRegister",
                   uGDSBufRegister(gpu_buffer, buffer_bytes, 0));
        buffer_registered = true;
        check_ugds("uGDSBatchIOSetUp",
                   uGDSBatchIOSetUp(&batch, handle, cfg.batch_size));
    }
};

Result run_phase(Context& ctx, const Config& cfg, double seconds,
                 ComputeWorker* worker, double baseline, uint64_t* cursor)
{
    constexpr uint64_t kRegionBase = 1ULL << 30;
    const uint64_t batch_bytes = cfg.io_size * cfg.batch_size;
    if (ctx.capacity <= kRegionBase)
        throw std::runtime_error("device is too small for benchmark region");
    uint64_t region_bytes = std::min<uint64_t>(4ULL << 30,
                                               ctx.capacity - kRegionBase);
    region_bytes -= region_bytes % 4096;
    if (region_bytes < batch_bytes)
        throw std::runtime_error("device is too small for benchmark region");

    std::vector<uGDSIOParams_t> params(cfg.batch_size);
    std::vector<uGDSIOEvents_t> events(cfg.batch_size);
    uint64_t batches = 0;
    uint64_t bytes = 0;
    uint64_t worker_before = worker ? worker->ops() : 0;
    uint64_t irq_before = irq_total();
    double cpu_before = thread_cpu_seconds();
    auto begin = Clock::now();
    auto deadline = begin + std::chrono::duration<double>(seconds);

    do {
        if (*cursor + batch_bytes > region_bytes) *cursor = 0;
        for (unsigned i = 0; i < cfg.batch_size; ++i) {
            params[i] = {};
            params[i].devPtr_base = ctx.gpu_buffer;
            params[i].devPtr_offset = static_cast<off_t>(i * cfg.io_size);
            params[i].file_offset = static_cast<off_t>(
                kRegionBase + *cursor + i * cfg.io_size);
            params[i].size = cfg.io_size;
            params[i].opcode = UGDS_READ;
            params[i].cookie = reinterpret_cast<void*>(
                static_cast<uintptr_t>(i + 1));
        }

        check_ugds("uGDSBatchIOSubmit",
                   uGDSBatchIOSubmit(ctx.batch, cfg.batch_size,
                                     params.data(), 0));
        unsigned nr = cfg.batch_size;
        check_ugds("uGDSBatchIOGetStatus",
                   uGDSBatchIOGetStatus(ctx.batch, cfg.batch_size, &nr,
                                        events.data(), nullptr));
        if (nr != cfg.batch_size)
            throw std::runtime_error("short batch completion");
        for (unsigned i = 0; i < nr; ++i) {
            if (events[i].status != UGDS_BATCH_COMPLETE ||
                events[i].ret != static_cast<ssize_t>(cfg.io_size))
                throw std::runtime_error("failed batch completion");
        }
        ++batches;
        bytes += batch_bytes;
        *cursor += batch_bytes;
    } while (Clock::now() < deadline);

    double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
    double cpu = thread_cpu_seconds() - cpu_before;
    uint64_t irqs = irq_total() - irq_before;
    uint64_t compute_ops = worker ? worker->ops() - worker_before : 0;

    Result result;
    result.gib_s = bytes / static_cast<double>(1ULL << 30) / elapsed;
    result.io_cpu_pct = 100.0 * cpu / elapsed;
    result.compute_available_pct = worker && baseline > 0
        ? 100.0 * (compute_ops / elapsed) / baseline : 0;
    result.irq_per_batch = batches ? static_cast<double>(irqs) / batches : 0;
    return result;
}

void print_result(const char* mode, const char* scenario, const Result& result)
{
    std::cout << std::left << std::setw(12) << mode
              << std::setw(29) << scenario << std::right
              << std::setw(11) << std::fixed << std::setprecision(2)
              << result.gib_s
              << std::setw(12) << result.io_cpu_pct;
    if (result.compute_available_pct > 0)
        std::cout << std::setw(16) << result.compute_available_pct;
    else
        std::cout << std::setw(16) << "-";
    if (std::string(mode) == "interrupt")
        std::cout << std::setw(12) << result.irq_per_batch;
    else
        std::cout << std::setw(12) << "-";
    std::cout << '\n';
}

void run_mode(const Config& cfg, bool interrupt, double baseline)
{
    Context ctx;
    ctx.open(cfg, interrupt);
    uint64_t cursor = 0;

    run_phase(ctx, cfg, std::min(0.5, cfg.seconds / 2), nullptr,
              baseline, &cursor);
    Result io_only = run_phase(ctx, cfg, cfg.seconds, nullptr,
                               baseline, &cursor);
    print_result(interrupt ? "interrupt" : "polling", "I/O only", io_only);

    ComputeWorker worker(cfg.cpu);
    worker.start();
    run_phase(ctx, cfg, std::min(0.5, cfg.seconds / 2), &worker,
              baseline, &cursor);
    Result contended = run_phase(ctx, cfg, cfg.seconds, &worker,
                                 baseline, &cursor);
    worker.stop();
    print_result(interrupt ? "interrupt" : "polling",
                 "I/O + same-core compute", contended);
}

Config parse_args(int argc, char** argv)
{
    if (argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " [device] [gpu] [cpu] [io_kib] [batch] [seconds]\n";
        std::exit(2);
    }
    Config cfg;
    if (argc > 1) cfg.device = argv[1];
    if (argc > 2) cfg.gpu = std::stoi(argv[2]);
    if (argc > 3) cfg.cpu = std::stoi(argv[3]);
    if (argc > 4) cfg.io_size = std::stoull(argv[4]) * 1024;
    if (argc > 5) cfg.batch_size = std::stoul(argv[5]);
    if (argc > 6) cfg.seconds = std::stod(argv[6]);
    if (cfg.io_size == 0 || cfg.io_size % 4096 != 0 ||
        cfg.batch_size == 0 || cfg.batch_size > 128 || cfg.seconds <= 0)
        throw std::runtime_error("invalid benchmark arguments");
    return cfg;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        Config cfg = parse_args(argc, argv);
        check_cuda("cudaSetDevice", cudaSetDevice(cfg.gpu));
        pin_current_thread(cfg.cpu);
        double baseline = compute_baseline(cfg.cpu);

        std::cout << "Batch interrupt demonstration: "
                  << cfg.io_size / 1024 << " KiB x " << cfg.batch_size
                  << ", " << cfg.seconds << " s per measurement\n\n"
                  << std::left << std::setw(12) << "Mode"
                  << std::setw(29) << "Scenario" << std::right
                  << std::setw(11) << "GiB/s"
                  << std::setw(12) << "IO CPU %"
                  << std::setw(16) << "Compute avail %"
                  << std::setw(12) << "IRQ/batch" << '\n';

        run_mode(cfg, false, baseline);
        run_mode(cfg, true, baseline);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_batch_interrupt: " << error.what() << '\n';
        return 1;
    }
}
