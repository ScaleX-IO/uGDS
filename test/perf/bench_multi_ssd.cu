/*
 * Measure two uGDS SSDs in three phases: each SSD alone, then both together.
 * Every worker performs synchronous sequential reads from a separate range;
 * no I/O striping or automatic device placement is involved.
 *
 * Example:
 *   cmake --build build --target bench_multi_ssd -j8
 *   ./build/bench_multi_ssd \
 *     -f /dev/ugds_drv0 -F /dev/ugds_drv1 -d 0 \
 *     -t 8 -n 1024 -s 4K -j
 *
 * Options:
 *   -f <device0>  first uGDS device (required)
 *   -F <device1>  second uGDS device (required, distinct from device0)
 *   -d <gpu_id>   GPU index (default: 0)
 *   -t <threads>  threads per SSD (default: 8; total concurrent threads: 16)
 *   -n <count>    sequential reads per worker per phase (default: 1024)
 *   -s <size>     I/O size in bytes, or with K/M/G suffix (default: 4K)
 *   -j            emit JSON instead of human-readable output
 */

#include "../functional/test_utils.h"
#include "bench_utils.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static constexpr size_t kDefaultIoSize = 4 * KB;
static constexpr unsigned kDefaultThreadsPerDevice = 8;

struct PhaseResult {
    uint64_t completed = 0;
    uint64_t worker_elapsed_ns = 0;
    uint64_t wall_ns = 0;
};

static uint64_t elapsed_ns(const struct timespec& start,
                           const struct timespec& end)
{
    return static_cast<uint64_t>(end.tv_sec - start.tv_sec) * 1000000000ULL +
           (end.tv_nsec - start.tv_nsec);
}

static bool multiply_size(size_t left, size_t right, size_t* product)
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        return false;
    *product = left * right;
    return true;
}

static void reset_worker(Worker* worker)
{
    worker->completed = 0;
    worker->elapsed_ns = 0;
    worker->failed = false;
    worker->failed_iteration = 0;
    worker->failed_result = 0;
}

static bool run_phase(const std::vector<Worker*>& workers, PhaseResult* result)
{
    std::vector<pthread_t> threads(workers.size());
    struct timespec start = {}, end = {};

    for (Worker* worker : workers) reset_worker(worker);
    clock_gettime(CLOCK_MONOTONIC, &start);
    size_t started = 0;
    for (; started < workers.size(); ++started) {
        if (pthread_create(&threads[started], nullptr, read_worker,
                           workers[started]) != 0)
            break;
    }
    for (size_t i = 0; i < started; ++i) pthread_join(threads[i], nullptr);
    clock_gettime(CLOCK_MONOTONIC, &end);
    result->wall_ns = elapsed_ns(start, end);
    result->completed = 0;
    result->worker_elapsed_ns = 0;

    bool success = started == workers.size();
    for (Worker* worker : workers) {
        report_worker_failure(*worker);
        result->completed += worker->completed;
        result->worker_elapsed_ns += worker->elapsed_ns;
        success = success && !worker->failed &&
                  worker->completed == worker->iterations;
    }
    return success;
}

static double bandwidth_mib_s(uint64_t completed, size_t io_size,
                              uint64_t duration_ns)
{
    if (duration_ns == 0) return 0.0;
    const double bytes = static_cast<double>(completed) * io_size;
    return bytes * 1000000000.0 / duration_ns / MB;
}

static double iops(uint64_t completed, uint64_t duration_ns)
{
    if (duration_ns == 0) return 0.0;
    return static_cast<double>(completed) * 1000000000.0 / duration_ns;
}

static double avg_latency_us(const PhaseResult& result)
{
    if (result.completed == 0) return 0.0;
    return static_cast<double>(result.worker_elapsed_ns) /
           result.completed / 1000.0;
}

static void print_device_result(const char* phase, const char* label,
                                const PhaseResult& result, size_t io_size)
{
    printf("%-16s %-8s ops=%-8llu bytes=%-12llu elapsed_ms=%-10.3f "
           "iops=%-12.2f bandwidth_mib_s=%-10.2f avg_us=%-.2f\n",
           phase, label,
           static_cast<unsigned long long>(result.completed),
           static_cast<unsigned long long>(result.completed * io_size),
           result.wall_ns / 1000000.0,
           iops(result.completed, result.wall_ns),
           bandwidth_mib_s(result.completed, io_size, result.wall_ns),
           avg_latency_us(result));
}

static bool parse_positive_unsigned(const char* value, unsigned* parsed)
{
    char* end = nullptr;
    unsigned long result = strtoul(value, &end, 10);
    if (!value[0] || *end != '\0' || result == 0 ||
        result > std::numeric_limits<unsigned>::max())
        return false;
    *parsed = static_cast<unsigned>(result);
    return true;
}

int main(int argc, char** argv)
{
    const char* first_path = nullptr;
    const char* second_path = nullptr;
    int gpu_id = 0;
    unsigned iterations = 1024;
    unsigned threads_per_device = kDefaultThreadsPerDevice;
    size_t io_size = kDefaultIoSize;
    bool json = false;
    int opt;
    while ((opt = getopt(argc, argv, "f:F:d:t:n:s:j")) != -1) {
        if (opt == 'f') first_path = optarg;
        else if (opt == 'F') second_path = optarg;
        else if (opt == 'd') gpu_id = atoi(optarg);
        else if (opt == 't' && !parse_positive_unsigned(optarg, &threads_per_device)) {
            fprintf(stderr, "Error: threads per device must be a positive integer\n");
            return 1;
        } else if (opt == 'n' && !parse_positive_unsigned(optarg, &iterations)) {
            fprintf(stderr, "Error: reads per worker must be a positive integer\n");
            return 1;
        } else if (opt == 's') {
            io_size = parse_size(optarg);
        } else if (opt == 'j') json = true;
        else return 1;
    }
    if (!first_path || !second_path || strcmp(first_path, second_path) == 0 ||
        io_size == 0 || io_size % kDefaultIoSize != 0) {
        fprintf(stderr, "Usage: %s -f <device0> -F <device1> [-d gpu_id] "
                "[-t threads_per_device] [-n reads_per_worker] [-s io_size] [-j]\n"
                "I/O size must be a non-zero multiple of 4 KiB.\n", argv[0]);
        return 1;
    }

    size_t per_thread_bytes = 0;
    size_t per_device_bytes = 0;
    size_t buffer_bytes = 0;
    if (!multiply_size(iterations, io_size, &per_thread_bytes) ||
        !multiply_size(threads_per_device, per_thread_bytes, &per_device_bytes) ||
        !multiply_size(threads_per_device, io_size, &buffer_bytes) ||
        per_device_bytes > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        fprintf(stderr, "Error: selected workload is too large\n");
        return 1;
    }

    cudaSetDevice(gpu_id);
    if (uGDSDriverOpen().err != UGDS_SUCCESS) return 1;
    uGDSHandle_t first = open_handle_at(first_path);
    uGDSHandle_t second = open_handle_at(second_path);
    if (!first || !second) {
        close_handle(first);
        close_handle(second);
        uGDSDriverClose();
        return 1;
    }

    uGDSStorage_t storage = nullptr;
    if (uGDSStorageCreate(&storage).err != UGDS_SUCCESS ||
        uGDSStorageAddHandle(storage, first).err != UGDS_SUCCESS ||
        uGDSStorageAddHandle(storage, second).err != UGDS_SUCCESS) {
        uGDSStorageDestroy(storage);
        close_handle(first);
        close_handle(second);
        uGDSDriverClose();
        return 1;
    }

    uint32_t device_count = 0;
    uint64_t capacity = 0;
    uint64_t first_capacity = 0;
    uint64_t second_capacity = 0;
    uGDSHandle_t storage_first = nullptr;
    uGDSHandle_t storage_second = nullptr;
    if (uGDSStorageGetDeviceCount(storage, &device_count).err != UGDS_SUCCESS ||
        uGDSStorageGetCapacity(storage, &capacity).err != UGDS_SUCCESS ||
        uGDSStorageGetDeviceCapacity(storage, 0, &first_capacity).err != UGDS_SUCCESS ||
        uGDSStorageGetDeviceCapacity(storage, 1, &second_capacity).err != UGDS_SUCCESS ||
        uGDSStorageGetHandle(storage, 0, &storage_first).err != UGDS_SUCCESS ||
        uGDSStorageGetHandle(storage, 1, &storage_second).err != UGDS_SUCCESS ||
        per_device_bytes > first_capacity || per_device_bytes > second_capacity) {
        fprintf(stderr, "Error: unable to query storage metadata or workload exceeds a device capacity\n");
        uGDSStorageDestroy(storage);
        close_handle(first);
        close_handle(second);
        uGDSDriverClose();
        return 1;
    }

    void* first_buffer = nullptr;
    void* second_buffer = nullptr;
    bool first_registered = false;
    bool second_registered = false;
    if (cudaMalloc(&first_buffer, buffer_bytes) == cudaSuccess &&
        cudaMalloc(&second_buffer, buffer_bytes) == cudaSuccess) {
        first_registered =
            uGDSBufRegister(first_buffer, buffer_bytes, TEST_BUF_FLAGS).err == UGDS_SUCCESS;
        second_registered =
            uGDSBufRegister(second_buffer, buffer_bytes, TEST_BUF_FLAGS).err == UGDS_SUCCESS;
    }
    if (!first_registered || !second_registered) {
        if (first_registered) uGDSBufDeregister(first_buffer);
        if (second_registered) uGDSBufDeregister(second_buffer);
        if (first_buffer) cudaFree(first_buffer);
        if (second_buffer) cudaFree(second_buffer);
        uGDSStorageDestroy(storage);
        close_handle(first);
        close_handle(second);
        uGDSDriverClose();
        return 1;
    }

    std::vector<Worker> first_workers(threads_per_device);
    std::vector<Worker> second_workers(threads_per_device);
    std::vector<Worker*> first_phase;
    std::vector<Worker*> second_phase;
    std::vector<Worker*> concurrent_phase;
    first_phase.reserve(threads_per_device);
    second_phase.reserve(threads_per_device);
    concurrent_phase.reserve(threads_per_device * 2);
    for (unsigned i = 0; i < threads_per_device; ++i) {
        off_t offset = static_cast<off_t>(i) * static_cast<off_t>(per_thread_bytes);
        first_workers[i] = {storage_first,
                            static_cast<char*>(first_buffer) + i * io_size,
                            iterations, io_size, offset, 0, 0, "device0", false, 0, 0};
        second_workers[i] = {storage_second,
                             static_cast<char*>(second_buffer) + i * io_size,
                             iterations, io_size, offset, 0, 0, "device1", false, 0, 0};
        first_phase.push_back(&first_workers[i]);
        second_phase.push_back(&second_workers[i]);
        concurrent_phase.push_back(&first_workers[i]);
        concurrent_phase.push_back(&second_workers[i]);
    }

    PhaseResult single_first, single_second, concurrent_all, concurrent_first,
        concurrent_second;
    bool success = run_phase(first_phase, &single_first) &&
                   run_phase(second_phase, &single_second) &&
                   run_phase(concurrent_phase, &concurrent_all);
    for (const Worker& worker : first_workers) {
        concurrent_first.completed += worker.completed;
        concurrent_first.worker_elapsed_ns += worker.elapsed_ns;
    }
    for (const Worker& worker : second_workers) {
        concurrent_second.completed += worker.completed;
        concurrent_second.worker_elapsed_ns += worker.elapsed_ns;
    }
    concurrent_first.wall_ns = concurrent_all.wall_ns;
    concurrent_second.wall_ns = concurrent_all.wall_ns;

    if (!success) {
        uGDSBufDeregister(first_buffer);
        uGDSBufDeregister(second_buffer);
        cudaFree(first_buffer);
        cudaFree(second_buffer);
        uGDSStorageDestroy(storage);
        close_handle(first);
        close_handle(second);
        uGDSDriverClose();
        return 1;
    }

    const double first_bandwidth = bandwidth_mib_s(single_first.completed, io_size,
                                                    single_first.wall_ns);
    const double second_bandwidth = bandwidth_mib_s(single_second.completed, io_size,
                                                     single_second.wall_ns);
    const double concurrent_bandwidth = bandwidth_mib_s(concurrent_all.completed, io_size,
                                                         concurrent_all.wall_ns);
    const double theoretical_total_bandwidth = first_bandwidth + second_bandwidth;
    const double best_single_bandwidth = first_bandwidth > second_bandwidth ?
        first_bandwidth : second_bandwidth;
    const double concurrent_vs_theoretical_percent = theoretical_total_bandwidth > 0.0 ?
        concurrent_bandwidth * 100.0 / theoretical_total_bandwidth : 0.0;
    const double concurrent_vs_best_single_x = best_single_bandwidth > 0.0 ?
        concurrent_bandwidth / best_single_bandwidth : 0.0;
    const double concurrent_vs_best_single_percent =
        (concurrent_vs_best_single_x - 1.0) * 100.0;

    if (json) {
        printf("{\n");
        printf("  \"io_type\": \"synchronous read\",\n");
        printf("  \"access_pattern\": \"sequential\",\n");
        printf("  \"io_size_bytes\": %zu,\n", io_size);
        printf("  \"iterations_per_worker\": %u,\n", iterations);
        printf("  \"threads_per_device\": %u,\n", threads_per_device);
        printf("  \"concurrent_total_threads\": %u,\n", threads_per_device * 2);
        printf("  \"storage_device_count\": %u,\n", device_count);
        printf("  \"aggregate_capacity_bytes\": %llu,\n",
               static_cast<unsigned long long>(capacity));
        printf("  \"single_device0_bandwidth_mib_s\": %.4f,\n", first_bandwidth);
        printf("  \"single_device0_iops\": %.4f,\n",
               iops(single_first.completed, single_first.wall_ns));
        printf("  \"single_device0_avg_latency_us\": %.4f,\n",
               avg_latency_us(single_first));
        printf("  \"single_device1_bandwidth_mib_s\": %.4f,\n", second_bandwidth);
        printf("  \"single_device1_iops\": %.4f,\n",
               iops(single_second.completed, single_second.wall_ns));
        printf("  \"single_device1_avg_latency_us\": %.4f,\n",
               avg_latency_us(single_second));
        printf("  \"theoretical_total_bandwidth_mib_s\": %.4f,\n",
               theoretical_total_bandwidth);
        printf("  \"concurrent_wall_ms\": %.4f,\n",
               concurrent_all.wall_ns / 1000000.0);
        printf("  \"concurrent_device0_bandwidth_mib_s\": %.4f,\n",
               bandwidth_mib_s(concurrent_first.completed, io_size, concurrent_all.wall_ns));
        printf("  \"concurrent_device1_bandwidth_mib_s\": %.4f,\n",
               bandwidth_mib_s(concurrent_second.completed, io_size, concurrent_all.wall_ns));
        printf("  \"concurrent_aggregate_bandwidth_mib_s\": %.4f,\n",
               concurrent_bandwidth);
        printf("  \"concurrent_aggregate_iops\": %.4f,\n",
               iops(concurrent_all.completed, concurrent_all.wall_ns));
        printf("  \"concurrent_vs_theoretical_percent\": %.4f,\n",
               concurrent_vs_theoretical_percent);
        printf("  \"concurrent_vs_best_single_x\": %.4f,\n",
               concurrent_vs_best_single_x);
        printf("  \"concurrent_vs_best_single_percent\": %.4f\n",
               concurrent_vs_best_single_percent);
        printf("}\n");
    } else {
        printf("multi-SSD benchmark: io_type=synchronous_read access=sequential "
               "io_size_bytes=%zu threads_per_device=%u total_threads=%u "
               "iterations_per_worker=%u devices=%u capacity_bytes=%llu\n",
               io_size, threads_per_device, threads_per_device * 2, iterations,
               device_count, static_cast<unsigned long long>(capacity));
        print_device_result("single", "device0", single_first, io_size);
        print_device_result("single", "device1", single_second, io_size);
        print_device_result("concurrent", "device0", concurrent_first, io_size);
        print_device_result("concurrent", "device1", concurrent_second, io_size);
        printf("concurrent       aggregate ops=%llu elapsed_ms=%.3f "
               "bandwidth_mib_s=%.2f iops=%.2f\n",
               static_cast<unsigned long long>(concurrent_all.completed),
               concurrent_all.wall_ns / 1000000.0, concurrent_bandwidth,
               iops(concurrent_all.completed, concurrent_all.wall_ns));
        printf("relative: theoretical_total_bandwidth_mib_s=%.2f "
               "concurrent_vs_theoretical=%.2f%% "
               "concurrent_vs_best_single=%.2fx (%+.2f%%)\n",
               theoretical_total_bandwidth, concurrent_vs_theoretical_percent,
               concurrent_vs_best_single_x, concurrent_vs_best_single_percent);
    }

    uGDSBufDeregister(first_buffer);
    uGDSBufDeregister(second_buffer);
    cudaFree(first_buffer);
    cudaFree(second_buffer);
    uGDSStorageDestroy(storage);
    close_handle(first);
    close_handle(second);
    uGDSDriverClose();
    return 0;
}
