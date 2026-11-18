#include "test_host_io_utils.h"

#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <ugds_device> <scratch_byte_offset>\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr size_t io_size = 8UL * 1024 * 1024;
    TestDevice dev(argv[1], argv[2], io_size);
    const size_t page = dev.page_size;
    const size_t length = io_size + page + 1;
    void* buffer = nullptr;
    require(posix_memalign(&buffer, page, io_size + 2 * page) == 0, "posix_memalign");
    auto* data = static_cast<unsigned char*>(buffer) + page;
    require(uGDSBufRegister(data + 1, page, UGDS_REGISTER_HOST).err == UGDS_INVALID_VALUE,
            "unaligned registration");
    require(uGDSBufRegister(buffer, page, UGDS_REGISTER_HOST | UGDS_REGISTER_DMABUF).err ==
            UGDS_INVALID_VALUE, "mixed registration flags");
    check_status(uGDSBufRegister(buffer, length, UGDS_REGISTER_HOST), "host BufRegister");

    std::vector<unsigned char> expected(io_size);
    for (size_t i = 0; i < io_size; ++i)
        expected[i] = static_cast<unsigned char>(i * 37u + i / page);
    std::memcpy(data, expected.data(), io_size);
    check_io(uGDSWrite(dev.handle, buffer, io_size, dev.offset, page), io_size, "host write");
    std::memset(data, 0, io_size);
    check_io(uGDSRead(dev.handle, buffer, io_size, dev.offset, page), io_size, "host read");
    require(std::memcmp(data, expected.data(), io_size) == 0, "host sync data mismatch");
    require(uGDSRead(dev.handle, buffer, page, dev.offset, io_size + page) == -EINVAL,
            "sync read beyond registered byte length");
    require(uGDSRead(dev.handle, buffer, page, dev.offset, 1) == -EINVAL,
            "unaligned I/O buffer offset");

    size_t async_size = page;
    off_t async_offset = 0;
    ssize_t result = 0;
    require(uGDSReadAsync(dev.handle, buffer, &async_size, &dev.offset, &async_offset,
                          &result, nullptr).err == UGDS_IO_NOT_SUPPORTED, "host async rejected");

    uGDSBatchHandle_t batch = nullptr;
    check_status(uGDSBatchIOSetUp(&batch, dev.handle, 3), "BatchIOSetUp");
    uGDSIOParams_t io{buffer, dev.offset, static_cast<off_t>(io_size + page),
                      page, UGDS_READ, data};
    check_status(uGDSBatchIOSubmit(batch, 1, &io, 0), "BatchIOSubmit out of range");
    drain_batch(batch, &io, 1, true);

    io.size = io_size;
    io.devPtr_offset = page;
    io.opcode = UGDS_WRITE;
    for (auto& byte : expected)
        byte ^= 0xff;
    std::memcpy(data, expected.data(), io_size);
    check_status(uGDSBatchIOSubmit(batch, 1, &io, 0), "BatchIOSubmit host write");
    require(uGDSBufDeregister(buffer).err == UGDS_BUSY, "deregister before batch completion");
    drain_batch(batch, &io, 1);

    io.opcode = UGDS_READ;
    std::memset(data, 0, io_size);
    check_status(uGDSBatchIOSubmit(batch, 1, &io, 0), "BatchIOSubmit host read");
    drain_batch(batch, &io, 1);
    require(std::memcmp(data, expected.data(), io_size) == 0, "host batch data mismatch");
    uGDSBatchIODestroy(batch);

    check_status(uGDSBufDeregister(buffer), "host BufDeregister");
    std::free(buffer);
    std::puts("PASS");
}
