#ifndef __UGDS_BENCH_UTILS_H__
#define __UGDS_BENCH_UTILS_H__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <string>
#include <getopt.h>

#define KB (1UL << 10)
#define MB (1UL << 20)
#define GB (1UL << 30)

struct BenchOpts {
    const char* file_path = nullptr;
    size_t length = 1UL * GB;
    size_t io_size = 4 * KB;
    int gpu_id = 0;
    int num_threads = 1;
    int io_depth = 1;
    bool is_write = false;
    bool is_batch = false;
    bool json = false;
};

struct ThreadData {
    int thread_id;
    int fd;
    int mode;
    off_t offset;
    size_t size;
    size_t io_size;
    size_t depth;
    void* gpu_buffer;
    void** gpu_buffers;
    void* handler;
    struct timespec start_time;
    struct timespec end_time;
    long long total_io_time;
    unsigned long long io_operations;
    std::vector<uint64_t> latency_vec;
    int device_id;
};

static inline uint64_t ts_diff_ns(const struct timespec& a, const struct timespec& b) {
    return (b.tv_sec - a.tv_sec) * 1000000000ULL + (b.tv_nsec - a.tv_nsec);
}

static inline void get_percentile(std::vector<uint64_t>& vec, double p) {
    if (vec.empty()) return;
    size_t idx = (size_t)(p * vec.size());
    if (idx >= vec.size()) idx = vec.size() - 1;
    printf("  p%.1f: %.2f us\n", p * 100, vec[idx] / 1000.0);
}

static inline void report_results(const char* label, std::vector<ThreadData>& threads,
                                  size_t io_size,
                                  uint64_t prog_time_ns, bool json) {
    uint64_t total_ops = 0;
    long long total_io_time = 0;
    std::vector<uint64_t> all_lat;

    for (auto& t : threads) {
        total_ops += t.io_operations;
        total_io_time += t.total_io_time;
        all_lat.insert(all_lat.end(), t.latency_vec.begin(), t.latency_vec.end());
    }

    std::sort(all_lat.begin(), all_lat.end());

    double bw_mbps = (total_ops * io_size * 1.0 / MB) / (prog_time_ns / 1e9);
    double avg_lat_us = total_ops > 0 ? (double)total_io_time / (total_ops * 1000.0) : 0;

    if (json) {
        printf("{\n");
        printf("  \"label\": \"%s\",\n", label);
        printf("  \"io_size\": %zu,\n", io_size);
        printf("  \"threads\": %zu,\n", threads.size());
        printf("  \"total_ops\": %llu,\n", (unsigned long long)total_ops);
        printf("  \"bandwidth_mbps\": %.2f,\n", bw_mbps);
        printf("  \"avg_latency_us\": %.2f,\n", avg_lat_us);
        if (!all_lat.empty()) {
            printf("  \"p50_us\": %.2f,\n", all_lat[(size_t)(0.5 * all_lat.size())] / 1000.0);
            printf("  \"p95_us\": %.2f,\n", all_lat[(size_t)(0.95 * all_lat.size())] / 1000.0);
            printf("  \"p99_us\": %.2f,\n", all_lat[(size_t)(0.99 * all_lat.size())] / 1000.0);
            printf("  \"p999_us\": %.2f\n", all_lat[(size_t)(0.999 * all_lat.size())] / 1000.0);
        }
        printf("}\n");
    } else {
        printf("[%s]\n", label);
        printf("  Total IO operations: %llu\n", (unsigned long long)total_ops);
        printf("  Bandwidth: %.2f MB/s\n", bw_mbps);
        bool is_batched = (!all_lat.empty() && all_lat.size() != total_ops);
        if (is_batched) {
            printf("  Avg latency: %.2f us (amortized per-IO)\n", avg_lat_us);
            size_t n_batches = all_lat.size();
            double batch_avg_us = 0;
            for (auto& l : all_lat) batch_avg_us += l;
            batch_avg_us /= (n_batches * 1000.0);
            printf("  Avg batch latency: %.2f us (%llu batches)\n",
                   batch_avg_us, (unsigned long long)n_batches);
        } else {
            printf("  Avg latency: %.2f us\n", avg_lat_us);
        }
        if (!all_lat.empty()) {
            get_percentile(all_lat, 0.50);
            get_percentile(all_lat, 0.95);
            get_percentile(all_lat, 0.99);
            get_percentile(all_lat, 0.999);
        }
        printf("\n");
    }
}

static inline size_t parse_size(const char* str) {
    std::string s(str);
    char unit = s.back();
    size_t mul = 1;
    if (unit == 'K' || unit == 'k') mul = KB;
    else if (unit == 'M' || unit == 'm') mul = MB;
    else if (unit == 'G' || unit == 'g') mul = GB;
    else return (size_t)strtoull(str, nullptr, 10);
    s.pop_back();
    return (size_t)strtoull(s.c_str(), nullptr, 10) * mul;
}

static inline bool parse_bench_opts(int argc, char** argv, BenchOpts& opts) {
    static struct option long_options[] = {
        {"file",    required_argument, nullptr, 'f'},
        {"length",  required_argument, nullptr, 'l'},
        {"size",    required_argument, nullptr, 's'},
        {"device",  required_argument, nullptr, 'd'},
        {"threads", required_argument, nullptr, 't'},
        {"iodepth", required_argument, nullptr, 'i'},
        {"mode",    required_argument, nullptr, 'm'},
        {"json",    no_argument,       nullptr, 'j'},
        {"help",    no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "f:l:s:d:t:i:m:jh", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'f': opts.file_path = optarg; break;
        case 'l': opts.length = parse_size(optarg); break;
        case 's': opts.io_size = parse_size(optarg); break;
        case 'd': opts.gpu_id = atoi(optarg); break;
        case 't': opts.num_threads = atoi(optarg); break;
        case 'i': opts.io_depth = atoi(optarg); break;
        case 'm':
            if (strcmp(optarg, "write") == 0) { opts.is_write = true; }
            else if (strcmp(optarg, "batch-read") == 0) { opts.is_batch = true; opts.is_write = false; }
            else if (strcmp(optarg, "batch-write") == 0) { opts.is_batch = true; opts.is_write = true; }
            break;
        case 'j': opts.json = true; break;
        case 'h':
        default:
            fprintf(stderr, "Usage: %s -f <file> [-l length] [-s io_size] [-d gpu] [-t threads] [-i depth] [-m read|write] [-j]\n", argv[0]);
            return false;
        }
    }
    if (!opts.file_path) {
        fprintf(stderr, "Error: -f <file_path> is required\n");
        return false;
    }
    return true;
}

#endif
