#include "ugds_safetensors.h"

namespace {
namespace abi {

template <typename T>
bool valid_abi(const T* value, size_t v1_size) {
    return value != nullptr && value->struct_size >= v1_size &&
           value->abi_version == UGDS_SAFETENSORS_ABI_VERSION;
}

uGDSError_t status(uGDSOpError code) {
    return uGDSError_t{code, 0};
}

void reset_mapping(uGDSTensorMapping_t* mapping) {
    mapping->name = nullptr;
    mapping->name_length = 0;
    mapping->shard_index = 0;
    mapping->shard_name = nullptr;
    mapping->shard_path = nullptr;
    mapping->dtype = nullptr;
    mapping->shape = nullptr;
    mapping->rank = 0;
    mapping->file_offset = 0;
    mapping->nbytes = 0;
}

void reset_shard_info(uGDSTensorShardInfo_t* shard) {
    shard->name = nullptr;
    shard->source_path = nullptr;
    shard->source_fd = -1;
    shard->canonical_size = 0;
    shard->header_length = 0;
}

void reset_lba_plan(uGDSTensorLbaPlan_t* plan) {
    plan->io_begin_lba = 0;
    plan->io_lba_count = 0;
    plan->io_offset = 0;
    plan->io_size = 0;
    plan->payload_skip = 0;
    plan->payload_size = 0;
}

}  // namespace abi
}  // namespace

#if defined(UGDS_ENABLE_SAFETENSORS)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#if defined(_CUDA)
#include <cuda_runtime_api.h>
#endif
#if defined(_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr uint64_t kMaxJsonBytes = 100000000;
constexpr size_t kMaxJsonDepth = 32;
constexpr size_t kMaxNameBytes = 1U << 20;
constexpr size_t kMaxRank = 64;
constexpr size_t kMaxTensors = 1000000;
constexpr size_t kMaxShards = 4096;
constexpr size_t kMaxObjectItems = kMaxTensors + 1;

struct TensorEntry {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
    uint64_t file_offset = 0;
    uint64_t nbytes = 0;
    size_t shard_index = 0;
};

namespace io {

class OwnedFd {
public:
    OwnedFd() = default;
    explicit OwnedFd(int fd) : fd_(fd) {}
    ~OwnedFd() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    OwnedFd(const OwnedFd&) = delete;
    OwnedFd& operator=(const OwnedFd&) = delete;

    OwnedFd(OwnedFd&& other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }
    OwnedFd& operator=(OwnedFd&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) {
                close(fd_);
            }
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    int get() const { return fd_; }

private:
    int fd_ = -1;
};

}  // namespace io

struct ShardState {
    std::string name;
    std::string path;
    io::OwnedFd fd;
    uint64_t canonical_size = 0;
    uint64_t header_length = 0;
};

struct ParsedShard {
    ShardState state;
    std::vector<TensorEntry> tensors;
};

struct TensorMapState {
    std::vector<ShardState> shards;
    std::vector<TensorEntry> tensors;
};

struct RawObjectState {
    std::string shard_name;
    uint64_t base_lba = 0;
    uint64_t canonical_size = 0;
    uint64_t padded_size = 0;
    uint64_t header_length = 0;
};

struct LbaMapState {
    uint64_t lba_size = 0;
    uint64_t io_alignment = 0;
    uint64_t region_begin = 0;
    uint64_t region_end = 0;
    uint64_t capacity_bytes = 0;
    std::vector<RawObjectState> objects;
};

namespace abi {

struct MappingError : std::runtime_error {
    MappingError(uGDSOpError code, const std::string& message)
        : std::runtime_error(message), status(code) {}

    uGDSOpError status;
};

[[noreturn]] void fail(uGDSOpError status, const std::string& message) {
    throw MappingError(status, message);
}

}  // namespace abi

namespace gpu {

constexpr uintptr_t kRegistrationAlignment = 1U << 16;

bool supports(int flags) {
    if (flags == 0) {
#if defined(_CUDA)
        return true;
#else
        return false;
#endif
    }
    if (flags == UGDS_REGISTER_DMABUF) {
#if defined(_HIP)
        return true;
#else
        return false;
#endif
    }
    return false;
}

uGDSError_t copy(int flags, void* destination, const void* source,
                 size_t size) {
    if (flags == UGDS_REGISTER_DMABUF) {
#if defined(_HIP)
        hipError_t error = hipMemcpy(
            destination, source, size, hipMemcpyDeviceToDevice);
        if (error == hipSuccess) {
            error = hipStreamSynchronize(nullptr);
        }
        return error == hipSuccess
            ? abi::status(UGDS_SUCCESS)
            : uGDSError_t{UGDS_CUDA_DRIVER_ERROR, static_cast<int>(error)};
#endif
    } else if (flags == 0) {
#if defined(_CUDA)
        cudaError_t error = cudaMemcpy(
            destination, source, size, cudaMemcpyDeviceToDevice);
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(nullptr);
        }
        return error == cudaSuccess
            ? abi::status(UGDS_SUCCESS)
            : uGDSError_t{UGDS_CUDA_DRIVER_ERROR, static_cast<int>(error)};
#endif
    }
    return abi::status(UGDS_IO_NOT_SUPPORTED);
}

}  // namespace gpu

namespace format {

uint64_t checked_add(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "integer addition overflow");
    }
    return left + right;
}

uint64_t checked_mul(uint64_t left, uint64_t right) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "integer multiplication overflow");
    }
    return left * right;
}

void validate_string_size(const std::string& value, const char* field) {
    if (value.size() > kMaxNameBytes) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  std::string("oversized ") + field);
    }
}

void validate_c_string(const std::string& value, const char* field) {
    validate_string_size(value, field);
    if (value.find('\0') != std::string::npos) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  std::string("invalid ") + field);
    }
}

json parse_json(
    const char* data, size_t size,
    uGDSOpError invalid_status = UGDS_SAFETENSORS_INVALID_FORMAT) {
    std::vector<std::unordered_set<std::string>> keys;
    std::vector<size_t> array_items;
    std::vector<bool> array_at_depth;

    auto count_array_item = [&](size_t depth) {
        if (depth < array_at_depth.size() && array_at_depth[depth] &&
            ++array_items[depth] > kMaxRank) {
            abi::fail(invalid_status, "JSON array item limit exceeded");
        }
    };

    auto strict_callback = [&](int depth, json::parse_event_t event,
                               json& parsed) {
        if (depth < 0 || static_cast<size_t>(depth) > kMaxJsonDepth) {
            abi::fail(invalid_status, "JSON nesting limit exceeded");
        }
        const size_t current_depth = static_cast<size_t>(depth);
        if (event == json::parse_event_t::value ||
            event == json::parse_event_t::object_start ||
            event == json::parse_event_t::array_start) {
            count_array_item(current_depth);
        }
        if (event == json::parse_event_t::object_start) {
            const size_t key_depth = current_depth + 1;
            if (key_depth > kMaxJsonDepth) {
                abi::fail(invalid_status, "JSON nesting limit exceeded");
            }
            if (keys.size() <= key_depth) {
                keys.resize(key_depth + 1);
            }
            keys[key_depth].clear();
        } else if (event == json::parse_event_t::key) {
            const size_t key_depth = current_depth;
            if (keys.size() <= key_depth) {
                keys.resize(key_depth + 1);
            }
            const auto& key = parsed.get_ref<const std::string&>();
            if (key.size() > kMaxNameBytes ||
                !keys[key_depth].insert(key).second) {
                abi::fail(invalid_status,
                          "invalid or duplicate JSON object key");
            }
            if (keys[key_depth].size() > kMaxObjectItems) {
                abi::fail(invalid_status,
                          "JSON object item limit exceeded");
            }
        } else if (event == json::parse_event_t::array_start) {
            const size_t array_depth = current_depth + 1;
            if (array_items.size() <= array_depth) {
                array_items.resize(array_depth + 1);
                array_at_depth.resize(array_depth + 1);
            }
            array_items[array_depth] = 0;
            array_at_depth[array_depth] = true;
        } else if (event == json::parse_event_t::array_end) {
            const size_t array_depth = current_depth + 1;
            if (array_depth < array_at_depth.size()) {
                array_at_depth[array_depth] = false;
            }
        }
        return true;
    };

    try {
        return json::parse(data, data + size, strict_callback, true, false);
    } catch (const abi::MappingError&) {
        throw;
    } catch (const json::exception&) {
        abi::fail(invalid_status, "invalid JSON");
    }
}

}  // namespace format

namespace io {

fs::path canonical_path(
    const fs::path& path,
    const char* error_message = "cannot resolve source path") {
    std::error_code ec;
    fs::path resolved = fs::canonical(path, ec);
    if (ec) {
        abi::fail(UGDS_SAFETENSORS_IO_ERROR, error_message);
    }
    return resolved;
}

fs::path absolute_path(const fs::path& path) {
    std::error_code ec;
    fs::path resolved = fs::absolute(path, ec);
    if (ec) {
        abi::fail(UGDS_SAFETENSORS_IO_ERROR,
                  "cannot resolve source path");
    }
    return resolved.lexically_normal();
}

OwnedFd open_readonly(const fs::path& path) {
    int fd;
    do {
        fd = open(path.c_str(),
                  O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        abi::fail(UGDS_SAFETENSORS_IO_ERROR,
                  "cannot open source file");
    }
    return OwnedFd(fd);
}

uint64_t source_file_size(int fd) {
    struct stat statbuf {};
    if (fstat(fd, &statbuf) != 0 || !S_ISREG(statbuf.st_mode) ||
        statbuf.st_size < 0) {
        abi::fail(UGDS_SAFETENSORS_IO_ERROR,
                  "cannot stat source file");
    }
    return static_cast<uint64_t>(statbuf.st_size);
}

uint64_t checked_offset(uint64_t offset, size_t delta) {
    const uint64_t amount = static_cast<uint64_t>(delta);
    if (amount > std::numeric_limits<uint64_t>::max() - offset) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "integer addition overflow");
    }
    return offset + amount;
}

void read_exact(int fd, void* buffer, size_t size, uint64_t offset) {
    size_t done = 0;
    while (done < size) {
        const uint64_t current = checked_offset(offset, done);
        if (current > static_cast<uint64_t>(
                          std::numeric_limits<off_t>::max())) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "file offset is too large");
        }
        ssize_t result = pread(fd, static_cast<char*>(buffer) + done,
                               size - done, static_cast<off_t>(current));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            abi::fail(UGDS_SAFETENSORS_IO_ERROR,
                      "cannot read source file");
        }
        done += static_cast<size_t>(result);
    }
}

std::string read_json_file(const fs::path& path) {
    OwnedFd fd = open_readonly(canonical_path(path));
    const uint64_t size = source_file_size(fd.get());
    if (size > kMaxJsonBytes) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "JSON file is too large");
    }

    std::string data(static_cast<size_t>(size), '\0');
    if (size != 0) {
        read_exact(fd.get(), data.data(), data.size(), 0);
    }
    return data;
}

}  // namespace io

namespace format {

uint64_t nonnegative_u64(const json& value, const char* field) {
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer()) {
        const int64_t number = value.get<int64_t>();
        if (number >= 0) {
            return static_cast<uint64_t>(number);
        }
    }
    abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
              std::string(field) + " must be a non-negative integer");
}

uint32_t dtype_bits(const std::string& dtype) {
    static constexpr std::pair<std::string_view, uint32_t> supported[] = {
        {"BOOL", 8},         {"U8", 8},          {"I8", 8},
        {"F8_E5M2", 8},     {"F8_E4M3", 8},     {"F8_E8M0", 8},
        {"F8_E4M3FNUZ", 8}, {"F8_E5M2FNUZ", 8}, {"I16", 16},
        {"U16", 16},        {"F16", 16},        {"BF16", 16},
        {"I32", 32},        {"U32", 32},        {"F32", 32},
        {"I64", 64},        {"U64", 64},        {"F64", 64},
        {"C64", 64},
    };

    for (const auto& item : supported) {
        if (item.first == dtype) {
            return item.second;
        }
    }
    abi::fail(UGDS_SAFETENSORS_UNSUPPORTED_DTYPE,
              "unsupported or sub-byte dtype");
}

void validate_metadata(const json& value) {
    if (!value.is_object()) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "__metadata__ must be an object");
    }
    for (const auto& item : value.items()) {
        if (!item.value().is_string()) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "__metadata__ values must be strings");
        }
    }
}

ParsedShard parse_shard(const fs::path& source_path,
                        const std::string& logical_name) {
    const fs::path source = io::absolute_path(source_path);
    const fs::path path = io::canonical_path(source);
    ParsedShard shard;
    shard.state.name = logical_name.empty() ? source.filename().string()
                                            : logical_name;
    validate_c_string(shard.state.name, "shard name");
    shard.state.path = source.string();
    shard.state.fd = io::open_readonly(path);
    const uint64_t size = io::source_file_size(shard.state.fd.get());
    shard.state.canonical_size = size;
    if (size < sizeof(uint64_t)) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "safetensors header is missing");
    }

    std::array<uint8_t, sizeof(uint64_t)> prefix{};
    io::read_exact(shard.state.fd.get(), prefix.data(), prefix.size(), 0);

    uint64_t header_size = 0;
    for (size_t i = 0; i < prefix.size(); ++i) {
        header_size |= static_cast<uint64_t>(prefix[i]) << (i * 8);
    }
    if (header_size > kMaxJsonBytes) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "safetensors header is too large");
    }
    shard.state.header_length = header_size;

    const uint64_t data_base = checked_add(sizeof(uint64_t), header_size);
    if (data_base > size || header_size == 0) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "invalid safetensors header length");
    }

    std::string header(static_cast<size_t>(header_size), '\0');
    io::read_exact(shard.state.fd.get(), header.data(), header.size(),
                   sizeof(uint64_t));
    if (header.front() != '{') {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "safetensors header must begin with an object");
    }

    const json root = parse_json(header.data(), header.size());
    if (!root.is_object()) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "safetensors metadata must be an object");
    }

    shard.tensors.reserve(root.size());

    for (const auto& item : root.items()) {
        const std::string& name = item.key();
        if (name == "__metadata__") {
            validate_metadata(item.value());
            continue;
        }
        validate_string_size(name, "tensor name");
        if (shard.tensors.size() == kMaxTensors) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "tensor count limit exceeded");
        }

        const json& value = item.value();
        if (!value.is_object() || value.size() != 3 ||
            !value.contains("dtype") || !value.contains("shape") ||
            !value.contains("data_offsets")) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "invalid tensor metadata object");
        }
        if (!value.at("dtype").is_string() ||
            !value.at("shape").is_array() ||
            !value.at("data_offsets").is_array() ||
            value.at("data_offsets").size() != 2) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "invalid tensor metadata fields");
        }

        TensorEntry tensor;
        tensor.name = name;
        tensor.dtype = value.at("dtype").get<std::string>();
        validate_c_string(tensor.dtype, "dtype");
        const uint32_t bits = dtype_bits(tensor.dtype);

        const json& shape = value.at("shape");
        if (shape.size() > kMaxRank) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "tensor rank limit exceeded");
        }
        uint64_t elements = 1;
        tensor.shape.reserve(shape.size());
        for (const json& dimension : shape) {
            const uint64_t dim = nonnegative_u64(dimension, "shape dimension");
            elements = checked_mul(elements, dim);
            tensor.shape.push_back(dim);
        }

        const json& offsets = value.at("data_offsets");
        tensor.data_begin = nonnegative_u64(offsets.at(0), "data offset");
        tensor.data_end = nonnegative_u64(offsets.at(1), "data offset");
        if (tensor.data_end < tensor.data_begin) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "reversed tensor range");
        }
        tensor.nbytes = tensor.data_end - tensor.data_begin;

        const uint64_t expected_bits = checked_mul(elements, bits);
        if (expected_bits % 8 != 0 || tensor.nbytes != expected_bits / 8) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "tensor shape and range size do not match");
        }

        tensor.file_offset = checked_add(data_base, tensor.data_begin);
        const uint64_t file_end = checked_add(data_base, tensor.data_end);
        if (file_end > size) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "tensor range exceeds shard size");
        }
        shard.tensors.push_back(std::move(tensor));
    }

    std::vector<const TensorEntry*> by_offset;
    by_offset.reserve(shard.tensors.size());
    for (const TensorEntry& tensor : shard.tensors) {
        by_offset.push_back(&tensor);
    }
    std::sort(by_offset.begin(), by_offset.end(),
              [](const TensorEntry* left, const TensorEntry* right) {
                  if (left->data_begin != right->data_begin) {
                      return left->data_begin < right->data_begin;
                  }
                  if (left->data_end != right->data_end) {
                      return left->data_end < right->data_end;
                  }
                  return left->name < right->name;
              });

    uint64_t expected_begin = 0;
    for (const TensorEntry* tensor : by_offset) {
        if (tensor->data_begin != expected_begin) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "tensor ranges overlap or leave a hole");
        }
        expected_begin = tensor->data_end;
    }
    if (expected_begin != size - data_base) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "safetensors payload is not fully indexed");
    }

    return shard;
}

bool path_is_within(const fs::path& base, const fs::path& candidate) {
    auto base_it = base.begin();
    auto candidate_it = candidate.begin();
    while (base_it != base.end()) {
        if (candidate_it == candidate.end() || *base_it != *candidate_it) {
            return false;
        }
        ++base_it;
        ++candidate_it;
    }
    return true;
}

fs::path resolve_shard_path(const fs::path& base,
                            const std::string& shard_name) {
    validate_c_string(shard_name, "shard path");
    if (shard_name.empty() || shard_name.find('\\') != std::string::npos) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "invalid shard path");
    }

    const fs::path relative = fs::u8path(shard_name);
    if (relative.is_absolute() || relative.has_root_name() ||
        relative.has_root_directory()) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "shard path must be relative");
    }
    for (const fs::path& component : relative) {
        if (component == "." || component == "..") {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "shard path traversal is not allowed");
        }
    }

    const fs::path candidate = (base / relative).lexically_normal();
    if (!path_is_within(base, candidate)) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "shard path escapes the index directory");
    }
    /* Keep snapshot-relative semantics; parse_shard follows cache symlinks. */
    return candidate;
}

}  // namespace format

namespace arithmetic {

uint64_t checked_add(uint64_t left, uint64_t right,
                     uGDSOpError status, const char* message) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        abi::fail(status, message);
    }
    return left + right;
}

uint64_t checked_mul(uint64_t left, uint64_t right,
                     uGDSOpError status, const char* message) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        abi::fail(status, message);
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

uint64_t checked_lcm(uint64_t left, uint64_t right,
                     uGDSOpError status) {
    if (left == 0 || right == 0) {
        abi::fail(status, "zero alignment");
    }
    return checked_mul(left / gcd(left, right), right, status,
                       "alignment overflow");
}

uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value - value % alignment;
}

uint64_t align_up(uint64_t value, uint64_t alignment,
                  uGDSOpError status) {
    const uint64_t remainder = value % alignment;
    return remainder == 0
               ? value
               : checked_add(value, alignment - remainder, status,
                             "aligned range overflow");
}

}  // namespace arithmetic

namespace manifest {

constexpr const char* kFormat = "ugds-safetensors-manifest";
constexpr uint64_t kVersion = 1;
constexpr uint64_t kObjectAlignmentFloor = 64U * 1024U;

[[noreturn]] void invalid(const std::string& message) {
    abi::fail(UGDS_SAFETENSORS_INVALID_MANIFEST, message);
}

const json& member(const json& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end()) {
        invalid(std::string("missing manifest field: ") + name);
    }
    return *found;
}

const json& object_member(const json& object, const char* name) {
    const json& value = member(object, name);
    if (!value.is_object()) {
        invalid(std::string("manifest field must be an object: ") + name);
    }
    return value;
}

uint64_t unsigned_member(const json& object, const char* name) {
    const json& value = member(object, name);
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer()) {
        const int64_t number = value.get<int64_t>();
        if (number >= 0) {
            return static_cast<uint64_t>(number);
        }
    }
    invalid(std::string("manifest field must be a non-negative integer: ") +
            name);
}

std::string string_member(const json& object, const char* name) {
    const json& value = member(object, name);
    if (!value.is_string()) {
        invalid(std::string("manifest field must be a string: ") + name);
    }
    std::string result = value.get<std::string>();
    if (result.size() > kMaxNameBytes ||
        result.find('\0') != std::string::npos) {
        invalid(std::string("invalid manifest string: ") + name);
    }
    return result;
}

bool power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_geometry(const uGDSTensorDeviceGeometry_t& geometry) {
    if (geometry.namespace_id == 0 || geometry.capacity_lbas == 0 ||
        !power_of_two(geometry.lba_size) ||
        !power_of_two(geometry.controller_page_size)) {
        abi::fail(UGDS_INVALID_VALUE, "invalid namespace geometry");
    }
}

std::unique_ptr<LbaMapState> load(
    const fs::path& path, const TensorMapState& primary,
    const uGDSTensorDeviceGeometry_t& geometry) {
    validate_geometry(geometry);
    const std::string data = io::read_json_file(path);
    const json root = format::parse_json(
        data.data(), data.size(), UGDS_SAFETENSORS_INVALID_MANIFEST);
    if (!root.is_object() || string_member(root, "format") != kFormat ||
        unsigned_member(root, "version") != kVersion) {
        invalid("unsupported raw-object manifest");
    }
    const json& device = object_member(root, "device");
    const uint64_t namespace_id = unsigned_member(device, "namespace_id");
    const uint64_t lba_size = unsigned_member(device, "lba_size");
    const uint64_t capacity_lbas =
        unsigned_member(device, "capacity_lbas");
    if (namespace_id != geometry.namespace_id ||
        lba_size != geometry.lba_size ||
        capacity_lbas != geometry.capacity_lbas) {
        abi::fail(UGDS_SAFETENSORS_DEVICE_MISMATCH,
                  "manifest namespace geometry does not match runtime");
    }

    const uint64_t capacity_bytes = arithmetic::checked_mul(
        geometry.capacity_lbas, geometry.lba_size,
        UGDS_SAFETENSORS_INVALID_MANIFEST, "namespace capacity overflow");
    const json& region = object_member(root, "region");
    const uint64_t region_begin = arithmetic::checked_mul(
        unsigned_member(region, "base_lba"), geometry.lba_size,
        UGDS_SAFETENSORS_INVALID_MANIFEST, "region offset overflow");
    const uint64_t region_size = arithmetic::checked_mul(
        unsigned_member(region, "length_lbas"), geometry.lba_size,
        UGDS_SAFETENSORS_INVALID_MANIFEST, "region size overflow");
    const uint64_t region_end = arithmetic::checked_add(
        region_begin, region_size, UGDS_SAFETENSORS_INVALID_MANIFEST,
        "region end overflow");
    if (region_size == 0 || region_end > capacity_bytes) {
        invalid("reserved region exceeds namespace capacity");
    }

    const uint64_t io_alignment = arithmetic::checked_lcm(
        geometry.lba_size, geometry.controller_page_size,
        UGDS_SAFETENSORS_INVALID_MANIFEST);
    const uint64_t object_alignment = arithmetic::checked_lcm(
        kObjectAlignmentFloor, io_alignment,
        UGDS_SAFETENSORS_INVALID_MANIFEST);

    const json& objects = object_member(root, "objects");
    if (objects.size() != primary.shards.size() ||
        objects.size() > kMaxShards) {
        invalid("manifest objects do not match primary shards");
    }

    std::unordered_map<std::string, RawObjectState> parsed;
    parsed.reserve(objects.size());
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(objects.size());
    for (const auto& item : objects.items()) {
        if (item.key().size() > kMaxNameBytes ||
            item.key().find('\0') != std::string::npos ||
            !item.value().is_object()) {
            invalid("invalid raw-object entry");
        }
        const json& value = item.value();
        RawObjectState object;
        object.shard_name = item.key();
        object.base_lba = unsigned_member(value, "base_lba");
        object.canonical_size =
            unsigned_member(value, "canonical_size");
        object.padded_size = unsigned_member(value, "padded_size");
        object.header_length = unsigned_member(value, "header_length");
        const uint64_t begin = arithmetic::checked_mul(
            object.base_lba, geometry.lba_size,
            UGDS_SAFETENSORS_INVALID_MANIFEST, "object offset overflow");
        const uint64_t end = arithmetic::checked_add(
            begin, object.padded_size,
            UGDS_SAFETENSORS_INVALID_MANIFEST, "object end overflow");
        if (object.canonical_size > object.padded_size ||
            begin % object_alignment != 0 ||
            object.padded_size % object_alignment != 0 ||
            begin < region_begin || end > region_end ||
            end > capacity_bytes) {
            invalid("raw object is unaligned or outside its reserved region");
        }
        ranges.emplace_back(begin, end);
        parsed.emplace(object.shard_name, std::move(object));
    }

    std::sort(ranges.begin(), ranges.end());
    for (size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].first < ranges[index - 1].second) {
            invalid("raw objects overlap");
        }
    }

    auto state = std::make_unique<LbaMapState>();
    state->lba_size = geometry.lba_size;
    state->io_alignment = io_alignment;
    state->region_begin = region_begin;
    state->region_end = region_end;
    state->capacity_bytes = capacity_bytes;
    state->objects.reserve(primary.shards.size());
    for (const ShardState& shard : primary.shards) {
        auto found = parsed.find(shard.name);
        if (found == parsed.end() ||
            found->second.canonical_size != shard.canonical_size ||
            found->second.header_length != shard.header_length) {
            invalid("manifest object identity does not match its shard");
        }
        state->objects.push_back(std::move(found->second));
    }
    return state;
}

}  // namespace manifest

namespace secondary {

void make_plan(const LbaMapState& state,
               const uGDSTensorMapping_t& tensor,
               uGDSTensorLbaPlan_t* plan) {
    abi::reset_lba_plan(plan);
    if (tensor.shard_index >= state.objects.size() ||
        tensor.shard_name == nullptr) {
        abi::fail(UGDS_INVALID_VALUE, "invalid tensor shard mapping");
    }
    const RawObjectState& object = state.objects[tensor.shard_index];
    if (object.shard_name != tensor.shard_name) {
        abi::fail(UGDS_INVALID_VALUE, "tensor shard identity mismatch");
    }
    const uint64_t file_end = arithmetic::checked_add(
        tensor.file_offset, tensor.nbytes, UGDS_INVALID_VALUE,
        "tensor file range overflow");
    if (file_end > object.canonical_size) {
        abi::fail(UGDS_INVALID_VALUE,
                  "tensor range exceeds its canonical shard");
    }
    plan->payload_size = tensor.nbytes;
    if (tensor.nbytes == 0) {
        return;
    }

    const uint64_t object_begin = arithmetic::checked_mul(
        object.base_lba, state.lba_size, UGDS_INTERNAL_ERROR,
        "validated object offset overflow");
    const uint64_t physical_begin = arithmetic::checked_add(
        object_begin, tensor.file_offset, UGDS_INVALID_VALUE,
        "tensor physical offset overflow");
    const uint64_t physical_end = arithmetic::checked_add(
        object_begin, file_end, UGDS_INVALID_VALUE,
        "tensor physical end overflow");
    const uint64_t io_begin =
        arithmetic::align_down(physical_begin, state.io_alignment);
    const uint64_t io_end = arithmetic::align_up(
        physical_end, state.io_alignment, UGDS_INVALID_VALUE);
    const uint64_t object_end = arithmetic::checked_add(
        object_begin, object.padded_size, UGDS_INTERNAL_ERROR,
        "validated object end overflow");
    const uint64_t io_size = io_end - io_begin;
    if (io_begin < object_begin || io_end > object_end ||
        io_begin < state.region_begin || io_end > state.region_end ||
        io_end > state.capacity_bytes ||
        io_end > static_cast<uint64_t>(
                     std::numeric_limits<off_t>::max()) ||
        io_size > std::numeric_limits<size_t>::max()) {
        abi::fail(UGDS_INVALID_VALUE,
                  "aligned tensor range exceeds the raw object");
    }

    plan->io_begin_lba = io_begin / state.lba_size;
    plan->io_lba_count = io_size / state.lba_size;
    plan->io_offset = io_begin;
    plan->io_size = io_size;
    plan->payload_skip = physical_begin - io_begin;
}

}  // namespace secondary

namespace primary {

std::unique_ptr<TensorMapState> build_single(const fs::path& path) {
    ParsedShard shard = format::parse_shard(path, "");
    auto state = std::make_unique<TensorMapState>();
    state->shards.push_back(std::move(shard.state));
    state->tensors = std::move(shard.tensors);
    for (TensorEntry& tensor : state->tensors) {
        tensor.shard_index = 0;
    }
    return state;
}

struct WeightRef {
    std::string name;
    std::string shard_name;
};

std::unique_ptr<TensorMapState> build_index(const fs::path& source_path) {
    const fs::path index_path = io::absolute_path(source_path);
    const std::string data = io::read_json_file(index_path);
    const json root = format::parse_json(data.data(), data.size());
    if (!root.is_object() || !root.contains("metadata") ||
        !root.at("metadata").is_object() ||
        !root.contains("weight_map") ||
        !root.at("weight_map").is_object()) {
        abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                  "index metadata and weight_map must be objects");
    }

    const fs::path base = io::canonical_path(
        index_path.parent_path(), "cannot resolve index directory");

    std::vector<WeightRef> weights;
    std::unordered_set<std::string> unique_shards;
    const json& weight_map = root.at("weight_map");
    weights.reserve(weight_map.size());
    for (const auto& item : weight_map.items()) {
        if (weights.size() == kMaxTensors || !item.value().is_string()) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "invalid index weight_map entry");
        }
        format::validate_string_size(item.key(), "tensor name");
        const std::string shard_name = item.value().get<std::string>();
        format::validate_c_string(shard_name, "shard path");
        if (unique_shards.insert(shard_name).second &&
            unique_shards.size() > kMaxShards) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "shard count limit exceeded");
        }
        weights.push_back(WeightRef{item.key(), shard_name});
    }

    std::vector<std::string> shard_names(unique_shards.begin(),
                                         unique_shards.end());
    std::sort(shard_names.begin(), shard_names.end());

    std::vector<ParsedShard> shards;
    shards.reserve(shard_names.size());
    for (const std::string& shard_name : shard_names) {
        const fs::path shard_path =
            format::resolve_shard_path(base, shard_name);
        shards.push_back(format::parse_shard(shard_path, shard_name));
    }

    struct ActualRef {
        size_t shard_index;
        size_t tensor_index;
    };
    std::unordered_map<std::string, ActualRef> actual;
    size_t tensor_count = 0;
    for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
        const ParsedShard& shard = shards[shard_index];
        if (shard.tensors.size() > kMaxTensors - tensor_count) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "tensor count limit exceeded");
        }
        tensor_count += shard.tensors.size();
        for (size_t tensor_index = 0;
             tensor_index < shard.tensors.size(); ++tensor_index) {
            const std::string& name = shard.tensors[tensor_index].name;
            if (!actual.emplace(name,
                                ActualRef{shard_index, tensor_index}).second) {
                abi::fail(UGDS_SAFETENSORS_INDEX_MISMATCH,
                          "tensor name appears in multiple shards");
            }
        }
    }

    if (actual.size() != weights.size()) {
        abi::fail(UGDS_SAFETENSORS_INDEX_MISMATCH,
                  "weight_map does not cover every shard tensor");
    }

    auto state = std::make_unique<TensorMapState>();
    state->shards.reserve(shards.size());
    for (ParsedShard& shard : shards) {
        state->shards.push_back(std::move(shard.state));
    }
    state->tensors.reserve(weights.size());
    for (const WeightRef& weight : weights) {
        const auto found = actual.find(weight.name);
        if (found == actual.end()) {
            abi::fail(UGDS_SAFETENSORS_INDEX_MISMATCH,
                      "weight_map tensor is missing from its shard");
        }
        const ActualRef ref = found->second;
        if (state->shards[ref.shard_index].name != weight.shard_name) {
            abi::fail(UGDS_SAFETENSORS_INDEX_MISMATCH,
                      "weight_map points to the wrong shard");
        }
        TensorEntry tensor = shards[ref.shard_index].tensors[ref.tensor_index];
        tensor.shard_index = ref.shard_index;
        state->tensors.push_back(std::move(tensor));
    }
    return state;
}

void finish_map(TensorMapState& state) {
    std::sort(state.tensors.begin(), state.tensors.end(),
              [](const TensorEntry& left, const TensorEntry& right) {
                  return left.name < right.name;
              });
    for (size_t index = 1; index < state.tensors.size(); ++index) {
        if (state.tensors[index - 1].name == state.tensors[index].name) {
            abi::fail(UGDS_SAFETENSORS_INVALID_FORMAT,
                      "duplicate tensor name");
        }
    }
}

void fill_mapping(const TensorMapState& state, const TensorEntry& tensor,
                  uGDSTensorMapping_t* mapping) {
    const ShardState& shard = state.shards[tensor.shard_index];
    abi::reset_mapping(mapping);
    mapping->name = tensor.name.data();
    mapping->name_length = tensor.name.size();
    mapping->shard_index = tensor.shard_index;
    mapping->shard_name = shard.name.c_str();
    mapping->shard_path = shard.path.c_str();
    mapping->dtype = tensor.dtype.c_str();
    mapping->shape = tensor.shape.data();
    mapping->rank = tensor.shape.size();
    mapping->file_offset = tensor.file_offset;
    mapping->nbytes = tensor.nbytes;
}

}  // namespace primary
}  // namespace

extern "C" uGDSError_t uGDSTensorMapOpen(
    uGDSTensorMap_t* map, const uGDSTensorMapDescr_t* descr) {
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *map = nullptr;
    if (!abi::valid_abi(descr, UGDS_TENSOR_MAP_DESCR_V1_SIZE) ||
        descr->path == nullptr ||
        descr->path[0] == '\0') {
        return abi::status(UGDS_INVALID_VALUE);
    }

    try {
        std::unique_ptr<TensorMapState> state;
        if (descr->type == UGDS_TENSOR_MAP_SINGLE_FILE) {
            state = primary::build_single(fs::u8path(descr->path));
        } else if (descr->type == UGDS_TENSOR_MAP_HF_INDEX) {
            state = primary::build_index(fs::u8path(descr->path));
        } else {
            return abi::status(UGDS_INVALID_VALUE);
        }
        primary::finish_map(*state);
        *map = state.release();
        return abi::status(UGDS_SUCCESS);
    } catch (const abi::MappingError& error) {
        return abi::status(error.status);
    } catch (const std::bad_alloc&) {
        return abi::status(UGDS_INTERNAL_ERROR);
    } catch (...) {
        return abi::status(UGDS_INTERNAL_ERROR);
    }
}

extern "C" void uGDSTensorMapClose(uGDSTensorMap_t map) {
    delete static_cast<TensorMapState*>(map);
}

extern "C" uGDSError_t uGDSTensorMapGetCount(uGDSTensorMap_t map,
                                               size_t* count) {
    if (count == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *count = 0;
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    const auto* state = static_cast<const TensorMapState*>(map);
    *count = state->tensors.size();
    return abi::status(UGDS_SUCCESS);
}

extern "C" uGDSError_t uGDSTensorMapFind(uGDSTensorMap_t map,
                                           const char* tensor_name,
                                           uGDSTensorMapping_t* mapping) {
    if (!abi::valid_abi(mapping, UGDS_TENSOR_MAPPING_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    if (tensor_name == nullptr) {
        abi::reset_mapping(mapping);
        return abi::status(UGDS_INVALID_VALUE);
    }
    return uGDSTensorMapFindN(map, tensor_name, std::strlen(tensor_name),
                              mapping);
}

extern "C" uGDSError_t uGDSTensorMapFindN(
    uGDSTensorMap_t map, const char* tensor_name,
    size_t tensor_name_length, uGDSTensorMapping_t* mapping) {
    if (!abi::valid_abi(mapping, UGDS_TENSOR_MAPPING_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_mapping(mapping);
    if (map == nullptr || tensor_name == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    const auto* state = static_cast<const TensorMapState*>(map);
    const std::string_view name(tensor_name, tensor_name_length);
    const auto found = std::lower_bound(
        state->tensors.begin(), state->tensors.end(), name,
        [](const TensorEntry& tensor, std::string_view value) {
            return std::string_view(tensor.name.data(), tensor.name.size()) <
                   value;
        });
    if (found == state->tensors.end() ||
        std::string_view(found->name.data(), found->name.size()) != name) {
        return abi::status(UGDS_SAFETENSORS_TENSOR_NOT_FOUND);
    }
    primary::fill_mapping(*state, *found, mapping);
    return abi::status(UGDS_SUCCESS);
}

extern "C" uGDSError_t uGDSTensorMapGetByIndex(
    uGDSTensorMap_t map, size_t index, uGDSTensorMapping_t* mapping) {
    if (!abi::valid_abi(mapping, UGDS_TENSOR_MAPPING_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_mapping(mapping);
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    const auto* state = static_cast<const TensorMapState*>(map);
    if (index >= state->tensors.size()) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    primary::fill_mapping(*state, state->tensors[index], mapping);
    return abi::status(UGDS_SUCCESS);
}

extern "C" uGDSError_t uGDSTensorMapGetShardCount(uGDSTensorMap_t map,
                                                    size_t* count) {
    if (count == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *count = 0;
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    const auto* state = static_cast<const TensorMapState*>(map);
    *count = state->shards.size();
    return abi::status(UGDS_SUCCESS);
}

extern "C" uGDSError_t uGDSTensorMapGetShardByIndex(
    uGDSTensorMap_t map, size_t index, uGDSTensorShardInfo_t* shard) {
    if (!abi::valid_abi(shard, UGDS_TENSOR_SHARD_INFO_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_shard_info(shard);
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    const auto* state = static_cast<const TensorMapState*>(map);
    if (index >= state->shards.size()) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    const ShardState& value = state->shards[index];
    shard->name = value.name.c_str();
    shard->source_path = value.path.c_str();
    shard->source_fd = value.fd.get();
    shard->canonical_size = value.canonical_size;
    shard->header_length = value.header_length;
    return abi::status(UGDS_SUCCESS);
}

extern "C" uGDSError_t uGDSTensorLbaMapOpen(
    uGDSTensorLbaMap_t* map, const uGDSTensorLbaMapDescr_t* descr) {
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *map = nullptr;
    if (!abi::valid_abi(descr, UGDS_TENSOR_LBA_MAP_DESCR_V1_SIZE) ||
        descr->manifest_path == nullptr || descr->manifest_path[0] == '\0' ||
        descr->tensor_map == nullptr ||
        !abi::valid_abi(descr->geometry,
                        UGDS_TENSOR_DEVICE_GEOMETRY_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }

    try {
        const auto* primary =
            static_cast<const TensorMapState*>(descr->tensor_map);
        std::unique_ptr<LbaMapState> state = manifest::load(
            fs::u8path(descr->manifest_path), *primary, *descr->geometry);
        *map = state.release();
        return abi::status(UGDS_SUCCESS);
    } catch (const abi::MappingError& error) {
        return abi::status(error.status);
    } catch (const std::bad_alloc&) {
        return abi::status(UGDS_INTERNAL_ERROR);
    } catch (...) {
        return abi::status(UGDS_INTERNAL_ERROR);
    }
}

extern "C" void uGDSTensorLbaMapClose(uGDSTensorLbaMap_t map) {
    delete static_cast<LbaMapState*>(map);
}

extern "C" uGDSError_t uGDSTensorLbaMapPlan(
    uGDSTensorLbaMap_t map, const uGDSTensorMapping_t* tensor,
    uGDSTensorLbaPlan_t* plan) {
    if (!abi::valid_abi(tensor, UGDS_TENSOR_MAPPING_V1_SIZE) ||
        !abi::valid_abi(plan, UGDS_TENSOR_LBA_PLAN_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_lba_plan(plan);
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    try {
        secondary::make_plan(*static_cast<const LbaMapState*>(map),
                             *tensor, plan);
        return abi::status(UGDS_SUCCESS);
    } catch (const abi::MappingError& error) {
        return abi::status(error.status);
    } catch (...) {
        return abi::status(UGDS_INTERNAL_ERROR);
    }
}

extern "C" uGDSError_t uGDSTensorReadInto(
    uGDSTensorLbaMap_t lba_map, const uGDSTensorMapping_t* tensor,
    const uGDSTensorReadDescr_t* read) {
    if (lba_map == nullptr ||
        !abi::valid_abi(tensor, UGDS_TENSOR_MAPPING_V1_SIZE) ||
        !abi::valid_abi(read, UGDS_TENSOR_READ_DESCR_V1_SIZE) ||
        read->io_handle == nullptr ||
        (read->staging_buffer_flags != 0 &&
         read->staging_buffer_flags != UGDS_REGISTER_DMABUF) ||
        tensor->nbytes > read->destination_size ||
        (tensor->nbytes != 0 && read->destination == nullptr)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    if (!gpu::supports(read->staging_buffer_flags)) {
        return abi::status(UGDS_IO_NOT_SUPPORTED);
    }

    uGDSTensorLbaPlan_t plan = UGDS_TENSOR_LBA_PLAN_INITIALIZER;
    const uGDSError_t status = uGDSTensorLbaMapPlan(
        lba_map, tensor, &plan);
    if (status.err != UGDS_SUCCESS) {
        return status;
    }
    if (plan.payload_size != tensor->nbytes) {
        return abi::status(UGDS_INTERNAL_ERROR);
    }
    if (tensor->nbytes == 0) {
        return plan.io_size == 0
            ? abi::status(UGDS_SUCCESS)
            : abi::status(UGDS_INTERNAL_ERROR);
    }
    if (plan.io_size == 0 ||
        plan.io_size > static_cast<uint64_t>(
                           std::numeric_limits<ssize_t>::max()) ||
        plan.payload_skip > plan.io_size ||
        plan.payload_size > plan.io_size - plan.payload_skip) {
        return abi::status(UGDS_INTERNAL_ERROR);
    }
    if (read->staging == nullptr ||
        reinterpret_cast<uintptr_t>(read->staging) %
                gpu::kRegistrationAlignment != 0 ||
        plan.io_size > read->staging_size) {
        return abi::status(UGDS_INVALID_VALUE);
    }

    const ssize_t bytes_read = uGDSRead(
        read->io_handle, read->staging,
        static_cast<size_t>(plan.io_size),
        static_cast<off_t>(plan.io_offset), 0);
    if (bytes_read != static_cast<ssize_t>(plan.io_size)) {
        // TODO(post-MVP): retain/poison timed-out I/O resources until drain.
        return abi::status(UGDS_SAFETENSORS_IO_ERROR);
    }

    const auto* payload = static_cast<const uint8_t*>(read->staging) +
                          static_cast<size_t>(plan.payload_skip);
    return gpu::copy(read->staging_buffer_flags, read->destination,
                     payload, static_cast<size_t>(plan.payload_size));
}

#else

namespace {
namespace abi {

uGDSError_t disabled_status() {
    return status(UGDS_IO_NOT_SUPPORTED);
}

}  // namespace abi
}  // namespace

extern "C" uGDSError_t uGDSTensorMapOpen(
    uGDSTensorMap_t* map, const uGDSTensorMapDescr_t* descr) {
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *map = nullptr;
    if (!abi::valid_abi(descr, UGDS_TENSOR_MAP_DESCR_V1_SIZE) ||
        descr->path == nullptr ||
        descr->path[0] == '\0') {
        return abi::status(UGDS_INVALID_VALUE);
    }
    return abi::disabled_status();
}

extern "C" void uGDSTensorMapClose(uGDSTensorMap_t) {}

extern "C" uGDSError_t uGDSTensorMapGetCount(uGDSTensorMap_t,
                                               size_t* count) {
    if (count == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *count = 0;
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorMapFind(uGDSTensorMap_t,
                                           const char* tensor_name,
                                           uGDSTensorMapping_t* mapping) {
    if (!abi::valid_abi(mapping, UGDS_TENSOR_MAPPING_V1_SIZE) ||
        tensor_name == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_mapping(mapping);
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorMapFindN(uGDSTensorMap_t,
                                            const char* tensor_name, size_t,
                                            uGDSTensorMapping_t* mapping) {
    if (!abi::valid_abi(mapping, UGDS_TENSOR_MAPPING_V1_SIZE) ||
        tensor_name == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_mapping(mapping);
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorMapGetByIndex(
    uGDSTensorMap_t, size_t, uGDSTensorMapping_t* mapping) {
    if (!abi::valid_abi(mapping, UGDS_TENSOR_MAPPING_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_mapping(mapping);
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorMapGetShardCount(uGDSTensorMap_t,
                                                    size_t* count) {
    if (count == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *count = 0;
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorMapGetShardByIndex(
    uGDSTensorMap_t, size_t, uGDSTensorShardInfo_t* shard) {
    if (!abi::valid_abi(shard, UGDS_TENSOR_SHARD_INFO_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_shard_info(shard);
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorLbaMapOpen(
    uGDSTensorLbaMap_t* map, const uGDSTensorLbaMapDescr_t* descr) {
    if (map == nullptr) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    *map = nullptr;
    if (!abi::valid_abi(descr, UGDS_TENSOR_LBA_MAP_DESCR_V1_SIZE) ||
        descr->manifest_path == nullptr || descr->manifest_path[0] == '\0' ||
        descr->tensor_map == nullptr ||
        !abi::valid_abi(descr->geometry,
                        UGDS_TENSOR_DEVICE_GEOMETRY_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    return abi::disabled_status();
}

extern "C" void uGDSTensorLbaMapClose(uGDSTensorLbaMap_t) {}

extern "C" uGDSError_t uGDSTensorLbaMapPlan(
    uGDSTensorLbaMap_t, const uGDSTensorMapping_t* tensor,
    uGDSTensorLbaPlan_t* plan) {
    if (!abi::valid_abi(tensor, UGDS_TENSOR_MAPPING_V1_SIZE) ||
        !abi::valid_abi(plan, UGDS_TENSOR_LBA_PLAN_V1_SIZE)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    abi::reset_lba_plan(plan);
    return abi::disabled_status();
}

extern "C" uGDSError_t uGDSTensorReadInto(
    uGDSTensorLbaMap_t lba_map, const uGDSTensorMapping_t* tensor,
    const uGDSTensorReadDescr_t* read) {
    if (lba_map == nullptr ||
        !abi::valid_abi(tensor, UGDS_TENSOR_MAPPING_V1_SIZE) ||
        !abi::valid_abi(read, UGDS_TENSOR_READ_DESCR_V1_SIZE) ||
        read->io_handle == nullptr ||
        (read->staging_buffer_flags != 0 &&
         read->staging_buffer_flags != UGDS_REGISTER_DMABUF) ||
        tensor->nbytes > read->destination_size ||
        (tensor->nbytes != 0 && read->destination == nullptr)) {
        return abi::status(UGDS_INVALID_VALUE);
    }
    return abi::disabled_status();
}

#endif
