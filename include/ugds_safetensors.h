#ifndef UGDS_SAFETENSORS_H_
#define UGDS_SAFETENSORS_H_

#include <stddef.h>
#include <stdint.h>

#include <ugds.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define UGDS_SAFETENSORS_API __attribute__((visibility("default")))
#else
#define UGDS_SAFETENSORS_API
#endif

#define UGDS_SAFETENSORS_ABI_VERSION 1U

typedef void* uGDSTensorMap_t;

typedef enum uGDSTensorMapSourceType {
    UGDS_TENSOR_MAP_SINGLE_FILE = 1,
    UGDS_TENSOR_MAP_HF_INDEX    = 2,
} uGDSTensorMapSourceType_t;

/* Initialize every public struct with its matching initializer macro. */
typedef struct uGDSTensorMapDescr {
    size_t                     struct_size;
    uint32_t                   abi_version;
    uGDSTensorMapSourceType_t  type;
    const char*                path;
} uGDSTensorMapDescr_t;

#define UGDS_TENSOR_MAP_DESCR_V1_SIZE                              \
    (offsetof(uGDSTensorMapDescr_t, path) +                        \
     sizeof(((uGDSTensorMapDescr_t*)0)->path))
#define UGDS_TENSOR_MAP_DESCR_INITIALIZER                          \
    {sizeof(uGDSTensorMapDescr_t), UGDS_SAFETENSORS_ABI_VERSION,   \
     UGDS_TENSOR_MAP_SINGLE_FILE, NULL}

/* All pointer fields are borrowed from the map and remain valid until close. */
typedef struct uGDSTensorMapping {
    size_t          struct_size;
    uint32_t        abi_version;
    const char*     name;
    size_t          name_length;
    /* Index accepted by uGDSTensorMapGetShardByIndex(). */
    size_t          shard_index;
    const char*     shard_name;
    const char*     shard_path;
    const char*     dtype;
    const uint64_t* shape;
    size_t          rank;
    /* Byte range in the canonical shard, measured from the file start. */
    uint64_t        file_offset;
    uint64_t        nbytes;
} uGDSTensorMapping_t;

#define UGDS_TENSOR_MAPPING_V1_SIZE                                \
    (offsetof(uGDSTensorMapping_t, nbytes) +                       \
     sizeof(((uGDSTensorMapping_t*)0)->nbytes))
#define UGDS_TENSOR_MAPPING_INITIALIZER                             \
    {sizeof(uGDSTensorMapping_t), UGDS_SAFETENSORS_ABI_VERSION,    \
     NULL, 0, 0, NULL, NULL, NULL, NULL, 0, 0, 0}

typedef struct uGDSTensorShardInfo {
    size_t      struct_size;
    uint32_t    abi_version;
    const char* name;
    const char* source_path;
    int         source_fd;
    uint64_t    canonical_size;
    uint64_t    header_length;
} uGDSTensorShardInfo_t;

#define UGDS_TENSOR_SHARD_INFO_V1_SIZE                             \
    (offsetof(uGDSTensorShardInfo_t, header_length) +              \
     sizeof(((uGDSTensorShardInfo_t*)0)->header_length))
#define UGDS_TENSOR_SHARD_INFO_INITIALIZER                         \
    {sizeof(uGDSTensorShardInfo_t), UGDS_SAFETENSORS_ABI_VERSION, \
     NULL, NULL, -1, 0, 0}

UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapOpen(
    uGDSTensorMap_t* map, const uGDSTensorMapDescr_t* descr);

UGDS_SAFETENSORS_API void uGDSTensorMapClose(uGDSTensorMap_t map);

UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapGetCount(
    uGDSTensorMap_t map, size_t* count);

UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapFind(
    uGDSTensorMap_t map, const char* tensor_name,
    uGDSTensorMapping_t* mapping);

UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapFindN(
    uGDSTensorMap_t map, const char* tensor_name, size_t tensor_name_length,
    uGDSTensorMapping_t* mapping);

UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapGetByIndex(
    uGDSTensorMap_t map, size_t index, uGDSTensorMapping_t* mapping);

UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapGetShardCount(
    uGDSTensorMap_t map, size_t* count);

/*
 * source_fd is the parsed file identity and must be used for later import.
 * It is borrowed, read-only, and valid until uGDSTensorMapClose(); source_path
 * is diagnostic and may name a symlink.
 */
UGDS_SAFETENSORS_API uGDSError_t uGDSTensorMapGetShardByIndex(
    uGDSTensorMap_t map, size_t index, uGDSTensorShardInfo_t* shard);

#ifdef __cplusplus
}
#endif

#endif /* UGDS_SAFETENSORS_H_ */
