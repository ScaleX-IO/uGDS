#include <ugds_safetensors.h>

#include <array>
#include <exception>
#include <iostream>

#if defined(UGDS_TEST_SAFETENSORS_ENABLED)

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

namespace {

#define REQUIRE(condition)                                                   \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAIL " << __func__ << ':' << __LINE__ << ": "    \
                      << #condition << '\n';                                 \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define REQUIRE_STATUS(expression, expected)                                 \
    do {                                                                     \
        const uGDSError_t result = (expression);                             \
        if (result.err != (expected)) {                                      \
            std::cerr << "FAIL " << __func__ << ':' << __LINE__             \
                      << ": expected " << uGDS_status_error(expected)        \
                      << ", got " << uGDS_status_error(result.err) << '\n';  \
            return false;                                                    \
        }                                                                    \
    } while (0)

#if defined(UGDS_TEST_SAFETENSORS_ENABLED)

namespace fs = std::filesystem;

class TempDir {
public:
    TempDir() {
        char pattern[] = "/tmp/ugds-safetensors-test-XXXXXX";
        char* created = mkdtemp(pattern);
        if (created == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        path_ = created;
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

struct MapGuard {
    ~MapGuard() {
        if (map != nullptr) {
            uGDSTensorMapClose(map);
        }
    }

    uGDSTensorMap_t map = nullptr;
};

struct SafetensorsImage {
    std::vector<uint8_t> bytes;
    uint64_t header_length = 0;
    uint64_t data_base = 0;
};

SafetensorsImage make_safetensors(std::string header, size_t payload_size) {
    while (header.size() % 8 != 0) {
        header.push_back(' ');
    }

    SafetensorsImage image;
    image.header_length = header.size();
    image.data_base = 8 + image.header_length;
    image.bytes.reserve(static_cast<size_t>(image.data_base) + payload_size);
    for (size_t i = 0; i < 8; ++i) {
        image.bytes.push_back(static_cast<uint8_t>(
            (image.header_length >> (i * 8)) & 0xffU));
    }
    image.bytes.insert(image.bytes.end(), header.begin(), header.end());
    for (size_t i = 0; i < payload_size; ++i) {
        image.bytes.push_back(static_cast<uint8_t>(i));
    }
    return image;
}

void write_bytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("cannot write test fixture");
    }
}

SafetensorsImage write_safetensors(const fs::path& path,
                                   const std::string& header,
                                   size_t payload_size) {
    SafetensorsImage image = make_safetensors(header, payload_size);
    write_bytes(path, image.bytes);
    return image;
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    if (!output) {
        throw std::runtime_error("cannot write test index");
    }
}

uGDSError_t open_map(const fs::path& path, uGDSTensorMapSourceType_t type,
                     uGDSTensorMap_t* map) {
    const std::string path_string = path.string();
    uGDSTensorMapDescr_t descr = UGDS_TENSOR_MAP_DESCR_INITIALIZER;
    descr.type = type;
    descr.path = path_string.c_str();
    return uGDSTensorMapOpen(map, &descr);
}

bool test_single_file_and_api() {
    TempDir temp;
    const fs::path file = temp.path() / "model.safetensors";
    const SafetensorsImage image = write_safetensors(
        file,
        R"({"normal":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[8,12]},"empty":{"dtype":"U8","shape":[0,4],"data_offsets":[12,12]},"nul\u0000name":{"dtype":"U8","shape":[1],"data_offsets":[12,13]}})",
        13);

    MapGuard guard;
    REQUIRE_STATUS(open_map(file, UGDS_TENSOR_MAP_SINGLE_FILE, &guard.map),
                   UGDS_SUCCESS);

    size_t count = 0;
    REQUIRE_STATUS(uGDSTensorMapGetCount(guard.map, &count), UGDS_SUCCESS);
    REQUIRE(count == 4);

    uGDSTensorMapping_t mapping = UGDS_TENSOR_MAPPING_INITIALIZER;
    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "normal", &mapping),
                   UGDS_SUCCESS);
    REQUIRE(mapping.name_length == 6);
    REQUIRE(std::memcmp(mapping.name, "normal", 6) == 0);
    REQUIRE(std::strcmp(mapping.dtype, "F16") == 0);
    REQUIRE(mapping.rank == 2 && mapping.shape[0] == 2 &&
            mapping.shape[1] == 2);
    REQUIRE(mapping.file_offset == image.data_base);
    REQUIRE(mapping.nbytes == 8);
    REQUIRE(mapping.shard_index == 0);

    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "scalar", &mapping),
                   UGDS_SUCCESS);
    REQUIRE(mapping.rank == 0 && mapping.nbytes == 4);
    REQUIRE(mapping.file_offset == image.data_base + 8);

    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "empty", &mapping),
                   UGDS_SUCCESS);
    REQUIRE(mapping.rank == 2 && mapping.shape[0] == 0 &&
            mapping.shape[1] == 4);
    REQUIRE(mapping.nbytes == 0);

    const std::array<char, 8> nul_name = {
        'n', 'u', 'l', '\0', 'n', 'a', 'm', 'e'};
    REQUIRE_STATUS(uGDSTensorMapFindN(guard.map, nul_name.data(),
                                     nul_name.size(), &mapping),
                   UGDS_SUCCESS);
    REQUIRE(mapping.name_length == nul_name.size());
    REQUIRE(std::memcmp(mapping.name, nul_name.data(), nul_name.size()) == 0);
    REQUIRE(mapping.nbytes == 1);

    const std::array<std::string, 4> expected_names = {
        "empty", "normal", std::string("nul\0name", 8), "scalar"};
    for (size_t i = 0; i < expected_names.size(); ++i) {
        REQUIRE_STATUS(uGDSTensorMapGetByIndex(guard.map, i, &mapping),
                       UGDS_SUCCESS);
        REQUIRE(mapping.name_length == expected_names[i].size());
        REQUIRE(std::memcmp(mapping.name, expected_names[i].data(),
                            mapping.name_length) == 0);
    }

    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "missing", &mapping),
                   UGDS_SAFETENSORS_TENSOR_NOT_FOUND);
    REQUIRE(mapping.name == nullptr && mapping.shape == nullptr &&
            mapping.nbytes == 0);

    uGDSTensorMapping_t bad_mapping = UGDS_TENSOR_MAPPING_INITIALIZER;
    bad_mapping.abi_version += 1;
    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "normal", &bad_mapping),
                   UGDS_INVALID_VALUE);

    size_t shard_count = 0;
    REQUIRE_STATUS(uGDSTensorMapGetShardCount(guard.map, &shard_count),
                   UGDS_SUCCESS);
    REQUIRE(shard_count == 1);

    uGDSTensorShardInfo_t shard = UGDS_TENSOR_SHARD_INFO_INITIALIZER;
    REQUIRE_STATUS(uGDSTensorMapGetShardByIndex(guard.map, 0, &shard),
                   UGDS_SUCCESS);
    REQUIRE(std::strcmp(shard.name, "model.safetensors") == 0);
    REQUIRE(fs::path(shard.source_path) == fs::absolute(file).lexically_normal());
    REQUIRE(shard.canonical_size == image.bytes.size());
    REQUIRE(shard.header_length == image.header_length);
    REQUIRE(shard.source_fd >= 0);
    struct stat statbuf {};
    REQUIRE(fstat(shard.source_fd, &statbuf) == 0 &&
            S_ISREG(statbuf.st_mode));

    const int retained_fd = shard.source_fd;
    uGDSTensorMapClose(guard.map);
    guard.map = nullptr;
    errno = 0;
    REQUIRE(fcntl(retained_fd, F_GETFD) == -1 && errno == EBADF);

    uGDSTensorMapDescr_t bad_descr = UGDS_TENSOR_MAP_DESCR_INITIALIZER;
    const std::string path_string = file.string();
    bad_descr.path = path_string.c_str();
    bad_descr.abi_version += 1;
    uGDSTensorMap_t bad_map = reinterpret_cast<void*>(1);
    REQUIRE_STATUS(uGDSTensorMapOpen(&bad_map, &bad_descr),
                   UGDS_INVALID_VALUE);
    REQUIRE(bad_map == nullptr);
    return true;
}

bool test_hf_shards_and_symlinks() {
    TempDir temp;
    const fs::path blobs = temp.path() / "blobs";
    const fs::path snapshot = temp.path() / "snapshot";
    fs::create_directories(blobs);
    fs::create_directories(snapshot);

    const std::string shard_a_name =
        "model-00001-of-00002.safetensors";
    const std::string shard_b_name =
        "model-00002-of-00002.safetensors";
    const SafetensorsImage shard_a = write_safetensors(
        blobs / "shard-a",
        R"({"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})",
        4);
    const SafetensorsImage shard_b = write_safetensors(
        blobs / "shard-b",
        R"({"b":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})",
        4);
    write_text(
        blobs / "index",
        std::string("{\"metadata\":{\"total_size\":8},\"weight_map\":{") +
            "\"a\":\"" + shard_a_name + "\",\"b\":\"" +
            shard_b_name + "\"}}");

    fs::create_symlink("../blobs/index",
                       snapshot / "model.safetensors.index.json");
    fs::create_symlink("../blobs/shard-a", snapshot / shard_a_name);
    fs::create_symlink("../blobs/shard-b", snapshot / shard_b_name);

    MapGuard guard;
    REQUIRE_STATUS(open_map(snapshot / "model.safetensors.index.json",
                            UGDS_TENSOR_MAP_HF_INDEX, &guard.map),
                   UGDS_SUCCESS);

    size_t tensor_count = 0;
    size_t shard_count = 0;
    REQUIRE_STATUS(uGDSTensorMapGetCount(guard.map, &tensor_count),
                   UGDS_SUCCESS);
    REQUIRE_STATUS(uGDSTensorMapGetShardCount(guard.map, &shard_count),
                   UGDS_SUCCESS);
    REQUIRE(tensor_count == 2 && shard_count == 2);

    uGDSTensorMapping_t tensor = UGDS_TENSOR_MAPPING_INITIALIZER;
    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "a", &tensor), UGDS_SUCCESS);
    REQUIRE(tensor.shard_index == 0);
    REQUIRE(std::strcmp(tensor.shard_name, shard_a_name.c_str()) == 0);
    REQUIRE(tensor.nbytes == 4);
    REQUIRE_STATUS(uGDSTensorMapFind(guard.map, "b", &tensor), UGDS_SUCCESS);
    REQUIRE(tensor.shard_index == 1);
    REQUIRE(std::strcmp(tensor.shard_name, shard_b_name.c_str()) == 0);

    uGDSTensorShardInfo_t info_a = UGDS_TENSOR_SHARD_INFO_INITIALIZER;
    uGDSTensorShardInfo_t info_b = UGDS_TENSOR_SHARD_INFO_INITIALIZER;
    REQUIRE_STATUS(uGDSTensorMapGetShardByIndex(guard.map, 0, &info_a),
                   UGDS_SUCCESS);
    REQUIRE_STATUS(uGDSTensorMapGetShardByIndex(guard.map, 1, &info_b),
                   UGDS_SUCCESS);
    REQUIRE(fs::path(info_a.source_path) ==
            fs::absolute(snapshot / shard_a_name).lexically_normal());
    REQUIRE(fs::path(info_b.source_path) ==
            fs::absolute(snapshot / shard_b_name).lexically_normal());
    REQUIRE(fs::is_symlink(info_a.source_path) &&
            fs::is_symlink(info_b.source_path));
    REQUIRE(info_a.header_length == shard_a.header_length);
    REQUIRE(info_b.header_length == shard_b.header_length);
    REQUIRE(info_a.canonical_size == shard_a.bytes.size());
    REQUIRE(info_b.canonical_size == shard_b.bytes.size());
    REQUIRE(info_a.source_fd >= 0 && info_b.source_fd >= 0 &&
            info_a.source_fd != info_b.source_fd);
    return true;
}

bool expect_single_error(const fs::path& path, uGDSOpError expected) {
    uGDSTensorMap_t map = reinterpret_cast<void*>(1);
    REQUIRE_STATUS(open_map(path, UGDS_TENSOR_MAP_SINGLE_FILE, &map),
                   expected);
    REQUIRE(map == nullptr);
    return true;
}

bool test_invalid_safetensors() {
    TempDir temp;
    struct BadCase {
        const char* name;
        std::vector<uint8_t> bytes;
        uGDSOpError expected;
    };

    std::vector<uint8_t> truncated(10, 0);
    truncated[0] = 64;
    truncated[8] = '{';
    truncated[9] = '}';

    const std::vector<BadCase> cases = {
        {"truncated", std::move(truncated),
         UGDS_SAFETENSORS_INVALID_FORMAT},
        {"duplicate-key",
         make_safetensors(
             R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
             1).bytes,
         UGDS_SAFETENSORS_INVALID_FORMAT},
        {"range-hole",
         make_safetensors(
             R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"y":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}})",
             3).bytes,
         UGDS_SAFETENSORS_INVALID_FORMAT},
        {"range-overlap",
         make_safetensors(
             R"({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
             2).bytes,
         UGDS_SAFETENSORS_INVALID_FORMAT},
        {"shape-size",
         make_safetensors(
             R"({"x":{"dtype":"F16","shape":[2],"data_offsets":[0,2]}})",
             2).bytes,
         UGDS_SAFETENSORS_INVALID_FORMAT},
        {"sub-byte-dtype",
         make_safetensors(
             R"({"x":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}})",
             1).bytes,
         UGDS_SAFETENSORS_UNSUPPORTED_DTYPE},
    };

    for (const BadCase& item : cases) {
        const fs::path file = temp.path() / item.name;
        write_bytes(file, item.bytes);
        REQUIRE(expect_single_error(file, item.expected));
    }
    return true;
}

bool expect_index_error(const fs::path& path, uGDSOpError expected) {
    uGDSTensorMap_t map = reinterpret_cast<void*>(1);
    REQUIRE_STATUS(open_map(path, UGDS_TENSOR_MAP_HF_INDEX, &map), expected);
    REQUIRE(map == nullptr);
    return true;
}

bool test_invalid_hf_index() {
    TempDir temp;

    const fs::path wrong = temp.path() / "wrong";
    fs::create_directory(wrong);
    write_safetensors(
        wrong / "one.safetensors",
        R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
        1);
    write_safetensors(
        wrong / "two.safetensors",
        R"({"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
        1);
    write_text(wrong / "index.json",
               R"({"metadata":{},"weight_map":{"a":"two.safetensors","b":"one.safetensors"}})");
    REQUIRE(expect_index_error(wrong / "index.json",
                               UGDS_SAFETENSORS_INDEX_MISMATCH));

    const fs::path missing = temp.path() / "missing";
    fs::create_directory(missing);
    write_safetensors(
        missing / "one.safetensors",
        R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"extra":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
        2);
    write_text(missing / "index.json",
               R"({"metadata":{},"weight_map":{"a":"one.safetensors"}})");
    REQUIRE(expect_index_error(missing / "index.json",
                               UGDS_SAFETENSORS_INDEX_MISMATCH));

    const fs::path traversal = temp.path() / "traversal";
    fs::create_directory(traversal);
    write_text(traversal / "index.json",
               R"({"metadata":{},"weight_map":{"a":"../outside.safetensors"}})");
    REQUIRE(expect_index_error(traversal / "index.json",
                               UGDS_SAFETENSORS_INVALID_FORMAT));
    return true;
}

#else

bool test_disabled_api() {
    uGDSTensorMapDescr_t descr = UGDS_TENSOR_MAP_DESCR_INITIALIZER;
    descr.path = "/not-opened.safetensors";
    uGDSTensorMap_t map = reinterpret_cast<void*>(1);
    REQUIRE_STATUS(uGDSTensorMapOpen(&map, &descr), UGDS_IO_NOT_SUPPORTED);
    REQUIRE(map == nullptr);

    size_t count = 17;
    REQUIRE_STATUS(uGDSTensorMapGetCount(nullptr, &count),
                   UGDS_IO_NOT_SUPPORTED);
    REQUIRE(count == 0);

    size_t shard_count = 17;
    REQUIRE_STATUS(uGDSTensorMapGetShardCount(nullptr, &shard_count),
                   UGDS_IO_NOT_SUPPORTED);
    REQUIRE(shard_count == 0);

    uGDSTensorMapping_t tensor = UGDS_TENSOR_MAPPING_INITIALIZER;
    tensor.name = "stale";
    tensor.nbytes = 99;
    REQUIRE_STATUS(uGDSTensorMapFind(nullptr, "x", &tensor),
                   UGDS_IO_NOT_SUPPORTED);
    REQUIRE(tensor.name == nullptr && tensor.nbytes == 0);

    tensor.name = "stale";
    tensor.nbytes = 99;
    REQUIRE_STATUS(uGDSTensorMapFindN(nullptr, "x", 1, &tensor),
                   UGDS_IO_NOT_SUPPORTED);
    REQUIRE(tensor.name == nullptr && tensor.nbytes == 0);

    tensor.name = "stale";
    tensor.nbytes = 99;
    REQUIRE_STATUS(uGDSTensorMapGetByIndex(nullptr, 0, &tensor),
                   UGDS_IO_NOT_SUPPORTED);
    REQUIRE(tensor.name == nullptr && tensor.nbytes == 0);

    uGDSTensorShardInfo_t shard = UGDS_TENSOR_SHARD_INFO_INITIALIZER;
    shard.source_fd = 42;
    REQUIRE_STATUS(uGDSTensorMapGetShardByIndex(nullptr, 0, &shard),
                   UGDS_IO_NOT_SUPPORTED);
    REQUIRE(shard.source_fd == -1);

    tensor.abi_version += 1;
    REQUIRE_STATUS(uGDSTensorMapFind(nullptr, "x", &tensor),
                   UGDS_INVALID_VALUE);

    uGDSTensorMapClose(nullptr);
    return true;
}

#endif

struct NamedTest {
    const char* name;
    bool (*run)();
};

}  // namespace

int main() {
#if defined(UGDS_TEST_SAFETENSORS_ENABLED)
    const std::array<NamedTest, 4> tests = {{
        {"single-file-and-api", test_single_file_and_api},
        {"hf-shards-and-symlinks", test_hf_shards_and_symlinks},
        {"invalid-safetensors", test_invalid_safetensors},
        {"invalid-hf-index", test_invalid_hf_index},
    }};
#else
    const std::array<NamedTest, 1> tests = {{
        {"disabled-api", test_disabled_api},
    }};
#endif

    try {
        for (const NamedTest& test : tests) {
            if (!test.run()) {
                return 1;
            }
            std::cout << "PASS " << test.name << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL exception: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
