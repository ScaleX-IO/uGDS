#include <ugds_safetensors.h>
#include <nlohmann/json.hpp>
#include <linux/fs.h>
#include <linux/nvme_ioctl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <getopt.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace {
using json = nlohmann::json;
constexpr uint64_t kObjectAlignmentFloor = 64U * 1024U;
constexpr size_t kChunkSize = 64U * 1024U * 1024U;
struct Options {
    std::string checkpoint;
    std::string target;
    std::string manifest;
    uint64_t base_lba = 0;
    bool base_lba_set = false;
    bool commit = false;
};
struct Fd {
    int value = -1;
    ~Fd() {
        if (value >= 0) close(value);
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd() = default;
    explicit Fd(int fd) : value(fd) {}
    void close_checked() {
        const int fd = value;
        value = -1;
        if (close(fd) != 0) {
            throw std::runtime_error(std::string("close: ") +
                                     std::strerror(errno));
        }
    }
};
struct Map {
    uGDSTensorMap_t value = nullptr;
    ~Map() {
        if (value != nullptr) uGDSTensorMapClose(value);
    }
};
struct TemporaryManifest {
    std::string path;
    bool published = false;
    ~TemporaryManifest() {
        if (!published && !path.empty()) unlink(path.c_str());
    }
};
struct FreeBuffer {
    void operator()(void* pointer) const { std::free(pointer); }
};
[[noreturn]] void fail_errno(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}
void check(uGDSError_t status, const char* operation) {
    if (status.err != UGDS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            uGDS_status_error(status.err) + " (" +
            std::to_string(static_cast<int>(status.err)) + ")");
    }
}
uint64_t add(uint64_t left, uint64_t right, const char* message) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error(message);
    }
    return left + right;
}
uint64_t multiply(uint64_t left, uint64_t right, const char* message) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::runtime_error(message);
    }
    return left * right;
}
uint64_t gcd(uint64_t left, uint64_t right) {
    while (right != 0) {
        const uint64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}
uint64_t lcm(uint64_t left, uint64_t right) {
    if (left == 0 || right == 0) {
        throw std::runtime_error("zero alignment");
    }
    return multiply(left / gcd(left, right), right, "alignment overflow");
}
uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : add(value, alignment - remainder,
                                        "padded size overflow");
}
uint64_t parse_u64(const char* text, const char* name) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}
void usage(const char* program) {
    std::fprintf(stderr,
                 "Usage: %s --checkpoint FILE --target BLOCK_DEVICE "
                 "--base-lba LBA --manifest FILE [--commit]\n",
                 program);
}
Options parse_options(int argc, char** argv) {
    Options result;
    static const option options[] = {
        {"checkpoint", required_argument, nullptr, 'f'},
        {"target", required_argument, nullptr, 't'},
        {"base-lba", required_argument, nullptr, 'b'},
        {"manifest", required_argument, nullptr, 'o'},
        {"commit", no_argument, nullptr, 'c'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    for (;;) {
        const int value = getopt_long(argc, argv, "f:t:b:o:ch", options,
                                      nullptr);
        if (value == -1) break;
        switch (value) {
        case 'f': result.checkpoint = optarg; break;
        case 't': result.target = optarg; break;
        case 'b':
            result.base_lba = parse_u64(optarg, "--base-lba");
            result.base_lba_set = true;
            break;
        case 'o': result.manifest = optarg; break;
        case 'c': result.commit = true; break;
        case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
        default: usage(argv[0]); throw std::runtime_error("invalid arguments");
        }
    }
    if (optind != argc || result.checkpoint.empty() || result.target.empty() ||
        result.manifest.empty() || !result.base_lba_set) {
        usage(argv[0]);
        throw std::runtime_error("missing required argument");
    }
    if (result.base_lba == 0) {
        throw std::runtime_error("--base-lba must not be zero");
    }
    return result;
}
void read_exact(int fd, void* buffer, size_t size, uint64_t offset) {
    size_t done = 0;
    while (done < size) {
        const uint64_t current = add(offset, done, "source offset overflow");
        if (current > static_cast<uint64_t>(
                          std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("source offset exceeds off_t");
        }
        const ssize_t result = pread(
            fd, static_cast<unsigned char*>(buffer) + done, size - done,
            static_cast<off_t>(current));
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) fail_errno("pread checkpoint");
        if (result == 0) {
            throw std::runtime_error("checkpoint changed or was truncated");
        }
        done += static_cast<size_t>(result);
    }
}
void write_exact(int fd, const void* buffer, size_t size, uint64_t offset,
                 uint64_t lba_size) {
    size_t done = 0;
    while (done < size) {
        const uint64_t current = add(offset, done, "target offset overflow");
        if (current > static_cast<uint64_t>(
                          std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("target offset exceeds off_t");
        }
        const ssize_t result = pwrite(
            fd, static_cast<const unsigned char*>(buffer) + done, size - done,
            static_cast<off_t>(current));
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) fail_errno("pwrite raw object");
        if (result == 0 || static_cast<uint64_t>(result) % lba_size != 0) {
            throw std::runtime_error("invalid short raw-object write");
        }
        done += static_cast<size_t>(result);
    }
}
void write_all(int fd, const std::string& contents) {
    size_t done = 0;
    while (done < contents.size()) {
        const ssize_t result = write(fd, contents.data() + done,
                                     contents.size() - done);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) fail_errno("write temporary manifest");
        if (result == 0) throw std::runtime_error("short manifest write");
        done += static_cast<size_t>(result);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        uGDSTensorMapDescr_t map_descr = UGDS_TENSOR_MAP_DESCR_INITIALIZER;
        map_descr.type = UGDS_TENSOR_MAP_SINGLE_FILE;
        map_descr.path = options.checkpoint.c_str();
        Map tensor_map;
        check(uGDSTensorMapOpen(&tensor_map.value, &map_descr),
              "uGDSTensorMapOpen");
        size_t shard_count = 0;
        check(uGDSTensorMapGetShardCount(tensor_map.value, &shard_count),
              "uGDSTensorMapGetShardCount");
        if (shard_count != 1) {
            throw std::runtime_error("checkpoint must contain one shard");
        }
        uGDSTensorShardInfo_t shard = UGDS_TENSOR_SHARD_INFO_INITIALIZER;
        check(uGDSTensorMapGetShardByIndex(tensor_map.value, 0, &shard),
              "uGDSTensorMapGetShardByIndex");
        if (shard.name == nullptr || shard.source_path == nullptr ||
            shard.source_fd < 0 || shard.canonical_size == 0) {
            throw std::runtime_error("invalid shard information");
        }
        int target_flags = options.commit ? O_RDWR | O_DIRECT | O_EXCL
                                          : O_RDONLY;
        Fd target(open(options.target.c_str(), target_flags | O_CLOEXEC));
        if (target.value < 0) fail_errno("open target");
        struct stat target_stat {};
        if (fstat(target.value, &target_stat) != 0) fail_errno("fstat target");
        if (!S_ISBLK(target_stat.st_mode)) {
            throw std::runtime_error("target is not a block device");
        }
        const std::string sysfs_device =
            "/sys/dev/block/" + std::to_string(major(target_stat.st_rdev)) +
            ":" + std::to_string(minor(target_stat.st_rdev));
        struct stat sysfs_stat {};
        if (stat(sysfs_device.c_str(), &sysfs_stat) != 0) {
            fail_errno("stat target sysfs entry");
        }
        struct stat partition_stat {};
        if (lstat((sysfs_device + "/partition").c_str(), &partition_stat) == 0) {
            throw std::runtime_error("target must be a whole NVMe namespace");
        }
        if (errno != ENOENT) fail_errno("lstat target partition marker");
        const int namespace_id = ioctl(target.value, NVME_IOCTL_ID);
        if (namespace_id < 0) fail_errno("NVME_IOCTL_ID");
        if (namespace_id != 1) {
            throw std::runtime_error(
                "uGDS MVP supports NVMe namespace id 1 only");
        }
        int lba_value = 0;
        uint64_t capacity_bytes = 0;
        if (ioctl(target.value, BLKSSZGET, &lba_value) != 0) {
            fail_errno("BLKSSZGET");
        }
        if (ioctl(target.value, BLKGETSIZE64, &capacity_bytes) != 0) {
            fail_errno("BLKGETSIZE64");
        }
        const long page_value = sysconf(_SC_PAGESIZE);
        const auto valid_power_of_two = [](uint64_t value) {
            return value != 0 && (value & (value - 1)) == 0;
        };
        if (lba_value <= 0 || page_value <= 0 ||
            !valid_power_of_two(static_cast<uint64_t>(lba_value)) ||
            !valid_power_of_two(static_cast<uint64_t>(page_value)) ||
            static_cast<uint64_t>(lba_value) > UINT32_MAX ||
            static_cast<uint64_t>(page_value) > UINT32_MAX ||
            capacity_bytes == 0 || capacity_bytes % lba_value != 0) {
            throw std::runtime_error("invalid block-device geometry");
        }
        const uint64_t lba_size = static_cast<uint64_t>(lba_value);
        const uint64_t page_size = static_cast<uint64_t>(page_value);
        const uint64_t capacity_lbas = capacity_bytes / lba_size;
        const uint64_t object_alignment = lcm(
            kObjectAlignmentFloor, lcm(lba_size, page_size));
        const uint64_t base_bytes = multiply(options.base_lba, lba_size,
                                             "base LBA overflow");
        const uint64_t padded_size = align_up(shard.canonical_size,
                                              object_alignment);
        const uint64_t object_end = add(base_bytes, padded_size,
                                        "raw-object end overflow");
        if (base_bytes % object_alignment != 0) {
            throw std::runtime_error("base LBA is not object-aligned");
        }
        if (object_end > capacity_bytes || object_end >
            static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("raw object exceeds target capacity");
        }
        if (object_alignment > kChunkSize ||
            kChunkSize % object_alignment != 0) {
            throw std::runtime_error("unsupported device alignment");
        }
        const uint64_t length_lbas = padded_size / lba_size;
        json manifest = {
            {"format", "ugds-safetensors-manifest"},
            {"version", 1},
            {"device", {{"namespace_id", namespace_id}, {"lba_size", lba_size},
                        {"capacity_lbas", capacity_lbas}}},
            {"region", {{"base_lba", options.base_lba},
                        {"length_lbas", length_lbas}}},
            {"objects", json::object()},
        };
        manifest["objects"][shard.name] = {
            {"base_lba", options.base_lba},
            {"canonical_size", shard.canonical_size},
            {"padded_size", padded_size},
            {"header_length", shard.header_length},
        };
        const std::string manifest_text = manifest.dump(2) + "\n";
        std::fprintf(stderr,
                     "%s: %s -> %s, LBA [%llu, %llu), bytes [%llu, %llu)\n",
                     options.commit ? "commit" : "dry-run", shard.source_path,
                     options.target.c_str(),
                     static_cast<unsigned long long>(options.base_lba),
                     static_cast<unsigned long long>(
                         add(options.base_lba, length_lbas, "LBA overflow")),
                     static_cast<unsigned long long>(base_bytes),
                     static_cast<unsigned long long>(object_end));
        std::fwrite(manifest_text.data(), 1, manifest_text.size(), stdout);
        if (!options.commit) return EXIT_SUCCESS;
        struct stat existing {};
        if (lstat(options.manifest.c_str(), &existing) == 0) {
            throw std::runtime_error("manifest already exists");
        }
        if (errno != ENOENT) fail_errno("lstat manifest");
        TemporaryManifest temporary{options.manifest + ".tmp.XXXXXX"};
        Fd manifest_fd(mkstemp(temporary.path.data()));
        if (manifest_fd.value < 0) fail_errno("mkstemp manifest");
        if (fcntl(manifest_fd.value, F_SETFD, FD_CLOEXEC) != 0) {
            fail_errno("fcntl manifest");
        }
        write_all(manifest_fd.value, manifest_text);
        if (fsync(manifest_fd.value) != 0) fail_errno("fsync manifest");
        manifest_fd.close_checked();
        uGDSTensorDeviceGeometry_t geometry =
            UGDS_TENSOR_DEVICE_GEOMETRY_INITIALIZER;
        geometry.namespace_id = static_cast<uint32_t>(namespace_id);
        geometry.lba_size = static_cast<uint32_t>(lba_size);
        geometry.controller_page_size = static_cast<uint32_t>(page_size);
        geometry.capacity_lbas = capacity_lbas;
        uGDSTensorLbaMapDescr_t lba_descr =
            UGDS_TENSOR_LBA_MAP_DESCR_INITIALIZER;
        lba_descr.manifest_path = temporary.path.c_str();
        lba_descr.tensor_map = tensor_map.value;
        lba_descr.geometry = &geometry;
        uGDSTensorLbaMap_t lba_map = nullptr;
        check(uGDSTensorLbaMapOpen(&lba_map, &lba_descr),
              "uGDSTensorLbaMapOpen(temp manifest)");
        uGDSTensorLbaMapClose(lba_map);
        void* raw_buffer = nullptr;
        const int alloc_error = posix_memalign(&raw_buffer,
                                               object_alignment, kChunkSize);
        if (alloc_error != 0) {
            throw std::runtime_error(std::string("posix_memalign: ") +
                                     std::strerror(alloc_error));
        }
        std::unique_ptr<void, FreeBuffer> buffer(raw_buffer);
        for (uint64_t offset = 0; offset < padded_size;) {
            const size_t write_size = static_cast<size_t>(
                std::min<uint64_t>(kChunkSize, padded_size - offset));
            const size_t payload_size = offset < shard.canonical_size
                ? static_cast<size_t>(std::min<uint64_t>(
                      write_size, shard.canonical_size - offset))
                : 0;
            if (payload_size != 0) {
                read_exact(shard.source_fd, buffer.get(), payload_size, offset);
            }
            if (payload_size < write_size) {
                std::memset(static_cast<unsigned char*>(buffer.get()) +
                                payload_size,
                            0, write_size - payload_size);
            }
            write_exact(target.value, buffer.get(), write_size,
                        add(base_bytes, offset, "target offset overflow"),
                        lba_size);
            offset += write_size;
        }
        if (fdatasync(target.value) != 0) fail_errno("fdatasync target");
        if (link(temporary.path.c_str(), options.manifest.c_str()) != 0) {
            fail_errno("publish manifest");
        }
        if (unlink(temporary.path.c_str()) != 0)
            fail_errno("unlink temporary manifest");
        temporary.published = true;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ugds_import_safetensors: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
