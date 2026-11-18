#include <ugds.h>

#include <cuda_runtime.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr size_t io_size = 64 * 1024;

inline void require(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

inline void check_status(uGDSError_t status, const char* operation)
{
    if (status.err != UGDS_SUCCESS) {
        std::fprintf(stderr, "FAIL: %s: %s\n", operation, uGDS_status_error(status.err));
        std::exit(EXIT_FAILURE);
    }
}

struct TestDevice {
    int fd;
    uGDSHandle_t handle = nullptr;
    off_t offset;
    size_t page_size;

    TestDevice(const char* path, const char* offset_arg, size_t scratch_bytes)
    {
        long page = sysconf(_SC_PAGESIZE);
        require(page > 0, "sysconf(_SC_PAGESIZE)");
        page_size = static_cast<size_t>(page);

        char* end = nullptr;
        errno = 0;
        long long value = std::strtoll(offset_arg, &end, 10);
        require(errno == 0 && end != offset_arg && *end == '\0' &&
                value >= 0 && value % page == 0 &&
                static_cast<uint64_t>(value) <= std::numeric_limits<off_t>::max() - scratch_bytes,
                "expected a page-aligned byte offset within off_t range");
        offset = static_cast<off_t>(value);

        fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            std::perror(path);
            std::exit(EXIT_FAILURE);
        }
        check_status(uGDSDriverOpen(), "DriverOpen");
        uGDSDescr_t descr{};
        descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = fd;
        check_status(uGDSHandleRegister(&handle, &descr), "HandleRegister");
        uint64_t capacity = 0;
        check_status(uGDSGetDeviceCapacity(handle, &capacity), "GetDeviceCapacity");
        require(static_cast<uint64_t>(offset) + scratch_bytes <= capacity,
                "scratch range exceeds namespace capacity");
    }

    ~TestDevice()
    {
        check_status(uGDSHandleDeregisterEx(handle, 5), "HandleDeregister");
        close(fd);
        check_status(uGDSDriverClose(), "DriverClose");
    }
};

inline void check_io(ssize_t result, size_t size, const char* operation)
{
    if (result != static_cast<ssize_t>(size)) {
        std::fprintf(stderr, "FAIL: %s returned %zd, expected %zu\n", operation, result, size);
        std::exit(EXIT_FAILURE);
    }
}

void drain_batch(uGDSBatchHandle_t batch, const uGDSIOParams_t* expected, unsigned count)
{
    uGDSIOEvents_t events[2]{};
    unsigned ready = count;
    timespec timeout{10, 0};
    check_status(uGDSBatchIOGetStatus(batch, count, &ready, events, &timeout), "BatchIOGetStatus");
    require(ready == count, "batch incomplete");
    bool seen[2]{};
    for (unsigned i = 0; i < ready; ++i) {
        unsigned request = 0;
        while (request < count && events[i].cookie != expected[request].cookie)
            ++request;
        require(request < count && !seen[request], "unexpected or duplicate batch cookie");
        seen[request] = true;
        require(events[i].status == UGDS_BATCH_COMPLETE, "wrong batch completion status");
        check_io(events[i].ret, expected[request].size, "batch completion");
    }
}

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class HostOp, class GpuOp>
void run_together(HostOp host_op, GpuOp gpu_op)
{
    std::promise<void> host_ready;
    auto ready = host_ready.get_future();
    std::promise<void> start;
    auto go = start.get_future();
    std::thread host([&] {
        host_ready.set_value();
        go.wait();
        host_op();
    });
    ready.wait();
    start.set_value();
    gpu_op();
    host.join();
}
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "Usage: %s <ugds_device> <scratch_byte_offset> [gpu_id]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int gpu_id = 0;
    if (argc == 4) {
        char* end = nullptr;
        long value = std::strtol(argv[3], &end, 10);
        require(end != argv[3] && *end == '\0' && value >= 0 &&
                value <= std::numeric_limits<int>::max(), "invalid GPU index");
        gpu_id = static_cast<int>(value);
    }
    check_cuda(cudaSetDevice(gpu_id), "cudaSetDevice");
    TestDevice dev(argv[1], argv[2], 2 * io_size);
    void* host_buf = nullptr;
    void* gpu_buf = nullptr;
    require(posix_memalign(&host_buf, dev.page_size, io_size + dev.page_size) == 0, "posix_memalign");
    check_cuda(cudaMalloc(&gpu_buf, io_size), "cudaMalloc");
    require(uGDSBufRegister(host_buf, io_size, UGDS_REGISTER_HOST | UGDS_REGISTER_DMABUF).err ==
            UGDS_INVALID_VALUE, "mixed registration flags");
    check_status(uGDSBufRegister(host_buf, io_size + 1, UGDS_REGISTER_HOST), "host BufRegister");
    check_status(uGDSBufRegister(gpu_buf, io_size, 0), "CUDA BufRegister");
    const off_t gpu_offset = dev.offset + io_size;

    require(uGDSRead(dev.handle, host_buf, dev.page_size, dev.offset, io_size) == -EINVAL,
            "host read beyond registered byte length");
    size_t async_size = io_size;
    off_t async_offset = 0;
    ssize_t async_result = 0;
    require(uGDSReadAsync(dev.handle, host_buf, &async_size, &dev.offset, &async_offset,
                         &async_result, nullptr).err == UGDS_IO_NOT_SUPPORTED,
            "host async rejected");

    const auto host_write = [&] {
        check_io(uGDSWrite(dev.handle, host_buf, io_size, dev.offset, 0), io_size, "host write");
    };
    const auto host_read = [&] {
        check_io(uGDSRead(dev.handle, host_buf, io_size, dev.offset, 0), io_size, "host read");
    };
    const auto gpu_write = [&] {
        check_io(uGDSWrite(dev.handle, gpu_buf, io_size, gpu_offset, 0), io_size, "GPU write");
    };
    const auto gpu_read = [&] {
        check_io(uGDSRead(dev.handle, gpu_buf, io_size, gpu_offset, 0), io_size, "GPU read");
    };

    std::vector<unsigned char> host_expected(io_size);
    std::vector<unsigned char> gpu_expected(io_size);
    std::vector<unsigned char> gpu_result(io_size);
    const auto fill = [&](unsigned seed) {
        for (size_t i = 0; i < io_size; ++i) {
            host_expected[i] = static_cast<unsigned char>(seed + i * 17u + i / dev.page_size);
            gpu_expected[i] = static_cast<unsigned char>(seed + 71u + i * 23u + i / dev.page_size);
        }
        std::memcpy(host_buf, host_expected.data(), io_size);
        check_cuda(cudaMemcpy(gpu_buf, gpu_expected.data(), io_size, cudaMemcpyHostToDevice),
                   "seed GPU buffer");
    };
    const auto clear_gpu = [&] {
        check_cuda(cudaMemset(gpu_buf, 0, io_size), "clear GPU buffer");
        check_cuda(cudaDeviceSynchronize(), "wait for clear");
    };
    const auto verify_host = [&] {
        require(std::memcmp(host_buf, host_expected.data(), io_size) == 0, "host data mismatch");
    };
    const auto verify_gpu = [&] {
        check_cuda(cudaMemcpy(gpu_result.data(), gpu_buf, io_size, cudaMemcpyDeviceToHost),
                   "inspect GPU buffer");
        require(gpu_result == gpu_expected, "GPU data mismatch");
    };

    fill(5);
    host_write();
    std::memset(host_buf, 0, io_size);
    host_read();
    verify_host();
    std::puts("PASS host-only sync");

    gpu_write();
    clear_gpu();
    gpu_read();
    verify_gpu();
    std::puts("PASS GPU-only sync");

    fill(33);
    run_together(host_write, gpu_write);
    std::memset(host_buf, 0, io_size);
    clear_gpu();
    run_together(host_read, gpu_read);
    verify_host();
    verify_gpu();
    std::puts("PASS mixed sync");

    fill(67);
    uGDSBatchHandle_t batch = nullptr;
    check_status(uGDSBatchIOSetUp(&batch, dev.handle, 4), "BatchIOSetUp");
    uGDSIOParams_t io[] = {
        {host_buf, dev.offset, 0, io_size, UGDS_WRITE, host_buf},
        {gpu_buf, gpu_offset, 0, io_size, UGDS_WRITE, gpu_buf},
    };
    check_status(uGDSBatchIOSubmit(batch, 2, io, 0), "BatchIOSubmit mixed write");
    require(uGDSBufDeregister(host_buf).err == UGDS_BUSY, "deregister before batch completion");
    drain_batch(batch, io, 2);
    std::memset(host_buf, 0, io_size);
    clear_gpu();
    for (auto& entry : io)
        entry.opcode = UGDS_READ;
    check_status(uGDSBatchIOSubmit(batch, 2, io, 0), "BatchIOSubmit mixed read");
    drain_batch(batch, io, 2);
    verify_host();
    verify_gpu();
    uGDSBatchIODestroy(batch);
    std::puts("PASS mixed batch");

    check_status(uGDSBufDeregister(gpu_buf), "GPU BufDeregister");
    check_status(uGDSBufDeregister(host_buf), "host BufDeregister");
    check_cuda(cudaFree(gpu_buf), "cudaFree");
    std::free(host_buf);
}
