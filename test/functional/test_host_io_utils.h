#ifndef UGDS_TEST_HOST_IO_UTILS_H
#define UGDS_TEST_HOST_IO_UTILS_H

#include <ugds.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <fcntl.h>
#include <unistd.h>

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

inline void drain_batch(uGDSBatchHandle_t batch, const uGDSIOParams_t* expected,
                        unsigned count, bool failed = false)
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
        require(events[i].status == (failed ? UGDS_BATCH_FAILED : UGDS_BATCH_COMPLETE),
                "wrong batch completion status");
        if (failed)
            require(events[i].ret == -EINVAL, "wrong batch error");
        else
            check_io(events[i].ret, expected[request].size, "batch completion");
    }
}

#endif
