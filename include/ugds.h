#ifndef __UGDS_H__
#define __UGDS_H__

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <time.h>

/* Buffer registration flags (defined locally to avoid pulling in
 * libnvm/nvm_dma.h, which transitively includes C++ <atomic>) */
#define NVM_MAP_DMABUF      0x1
#define NVM_MAP_RDMA        0x2    /* Retain dmabuf fd for RDMA use */
#define NVM_MAP_FORCE_CUDA  0x4    /* Force CUDA path (skip auto-probe) */
/* GPU runtime headers are NOT included in the public header.
 * cuda_runtime.h and hip_runtime_api.h define conflicting types
 * (vector types, stream types) that cannot coexist in a single TU.
 *
 * The async API uses void* for streams. Callers pass cudaStream_t
 * (CUDA) or hipStream_t (HIP), which implicitly convert to void*.
 * Backend-specific runtime headers are included only in internal
 * source files that need them. */

#ifdef __cplusplus
extern "C" {
#endif

#define UGDS_BASE_ERR 5000

typedef enum uGDSOpError {
    UGDS_SUCCESS                     = 0,
    UGDS_DRIVER_NOT_INITIALIZED      = UGDS_BASE_ERR + 1,
    UGDS_DRIVER_INVALID_PROPS        = UGDS_BASE_ERR + 2,
    UGDS_DRIVER_UNSUPPORTED_LIMIT    = UGDS_BASE_ERR + 3,
    UGDS_DRIVER_VERSION_MISMATCH     = UGDS_BASE_ERR + 4,
    UGDS_DRIVER_VERSION_READ_ERROR   = UGDS_BASE_ERR + 5,
    UGDS_DRIVER_CLOSING              = UGDS_BASE_ERR + 6,
    UGDS_PLATFORM_NOT_SUPPORTED      = UGDS_BASE_ERR + 7,
    UGDS_IO_NOT_SUPPORTED            = UGDS_BASE_ERR + 8,
    UGDS_DEVICE_NOT_SUPPORTED        = UGDS_BASE_ERR + 9,
    UGDS_NVFS_DRIVER_ERROR           = UGDS_BASE_ERR + 10,
    UGDS_CUDA_DRIVER_ERROR           = UGDS_BASE_ERR + 11,
    UGDS_CUDA_POINTER_INVALID        = UGDS_BASE_ERR + 12,
    UGDS_CUDA_MEMORY_TYPE_INVALID    = UGDS_BASE_ERR + 13,
    UGDS_CUDA_POINTER_RANGE_ERROR    = UGDS_BASE_ERR + 14,
    UGDS_CUDA_CONTEXT_MISMATCH       = UGDS_BASE_ERR + 15,
    UGDS_INVALID_MAPPING_SIZE        = UGDS_BASE_ERR + 16,
    UGDS_INVALID_MAPPING_RANGE       = UGDS_BASE_ERR + 17,
    UGDS_INVALID_FILE_TYPE           = UGDS_BASE_ERR + 18,
    UGDS_INVALID_FILE_OPEN_FLAG      = UGDS_BASE_ERR + 19,
    UGDS_DIO_NOT_SET                 = UGDS_BASE_ERR + 20,
    UGDS_INVALID_VALUE               = UGDS_BASE_ERR + 22,
    UGDS_MEMORY_ALREADY_REGISTERED   = UGDS_BASE_ERR + 23,
    UGDS_MEMORY_NOT_REGISTERED       = UGDS_BASE_ERR + 24,
    UGDS_PERMISSION_DENIED           = UGDS_BASE_ERR + 25,
    UGDS_DRIVER_ALREADY_OPEN         = UGDS_BASE_ERR + 26,
    UGDS_HANDLE_NOT_REGISTERED       = UGDS_BASE_ERR + 27,
    UGDS_HANDLE_ALREADY_REGISTERED   = UGDS_BASE_ERR + 28,
    UGDS_DEVICE_NOT_FOUND            = UGDS_BASE_ERR + 29,
    UGDS_INTERNAL_ERROR              = UGDS_BASE_ERR + 30,
    UGDS_GETNEWFD_FAILED             = UGDS_BASE_ERR + 31,
    UGDS_NVFS_SETUP_ERROR            = UGDS_BASE_ERR + 33,
    UGDS_IO_DISABLED                 = UGDS_BASE_ERR + 34,
    UGDS_GPU_MEMORY_PINNING_FAILED   = UGDS_BASE_ERR + 36,

    UGDS_BATCH_CAPACITY_EXCEEDED     = UGDS_BASE_ERR + 40,
    UGDS_BUSY                        = UGDS_BASE_ERR + 42,
} uGDSOpError;

static inline const char* uGDS_status_error(uGDSOpError status) {
    switch (status) {
    case UGDS_SUCCESS:                     return "success";
    case UGDS_DRIVER_NOT_INITIALIZED:      return "driver not initialized";
    case UGDS_DRIVER_INVALID_PROPS:        return "invalid property";
    case UGDS_DRIVER_UNSUPPORTED_LIMIT:    return "property range error";
    case UGDS_DRIVER_VERSION_MISMATCH:     return "driver version mismatch";
    case UGDS_DRIVER_CLOSING:              return "driver closing";
    case UGDS_IO_NOT_SUPPORTED:            return "IO not supported";
    case UGDS_INVALID_FILE_TYPE:           return "unsupported file type";
    case UGDS_INVALID_VALUE:               return "invalid arguments";
    case UGDS_MEMORY_ALREADY_REGISTERED:   return "memory already registered";
    case UGDS_MEMORY_NOT_REGISTERED:       return "memory not registered";
    case UGDS_INTERNAL_ERROR:              return "internal error";
    case UGDS_GPU_MEMORY_PINNING_FAILED:   return "GPU memory pinning failed";
    case UGDS_BATCH_CAPACITY_EXCEEDED:     return "batch capacity exceeded";

    case UGDS_BUSY:                        return "resource busy, retry";
    default:                                  return "unknown uGDS error";
    }
}

typedef struct uGDSError {
    uGDSOpError err;
    int           cu_err;
} uGDSError_t;

#define IS_UGDS_ERR(err)   (abs((err)) > UGDS_BASE_ERR)
#define UGDS_ERRSTR(err)   uGDS_status_error((uGDSOpError)abs((err)))

enum uGDSHandleType {
    UGDS_HANDLE_TYPE_OPAQUE_FD    = 1,
    UGDS_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    UGDS_HANDLE_TYPE_USERSPACE_FS = 3,
};

typedef struct uGDSDescr_t {
    enum uGDSHandleType type;
    union {
        int   fd;
        void* handle;
    } handle;
} uGDSDescr_t;

typedef void* uGDSHandle_t;

uGDSError_t uGDSDriverOpen(void);

uGDSError_t uGDSDriverClose(void);

uGDSError_t uGDSHandleRegister(uGDSHandle_t* fh, uGDSDescr_t* descr);

void uGDSHandleDeregister(uGDSHandle_t fh);

uGDSError_t uGDSBufRegister(const void* bufPtr_base, size_t length, int flags);

/* Flag for uGDSBufRegister: use AMD HIP/dma-buf path */
#define UGDS_REGISTER_DMABUF  NVM_MAP_DMABUF

uGDSError_t uGDSBufDeregister(const void* bufPtr_base);

/* ── dma-buf export support ── */

/* Extended buffer registration with explicit backend */
typedef enum uGDSBackend {
    UGDS_BACKEND_DEFAULT = 0,
    UGDS_BACKEND_CUDA    = 1,
    UGDS_BACKEND_HIP     = 2,
} uGDSBackend_t;

typedef struct uGDSBufConfig {
    uGDSBackend_t   backend;
    bool            enable_export;  /* request dma-buf fd for RDMA/peer export */
} uGDSBufConfig_t;

uGDSError_t uGDSBufRegisterEx(const void* bufPtr_base, size_t length,
                               const uGDSBufConfig_t* config);

/* dma-buf export handle for RDMA registration.
 * fd is dup()'d — caller owns it and MUST close after use. */
typedef struct uGDSDmabufExport {
    int       fd;
    uint64_t  offset;
    size_t    length;
} uGDSDmabufExport_t;

/* Export a dma-buf handle for external use (e.g., RDMA registration).
 * Returns UGDS_SUCCESS with 'out' filled.
 * Caller owns out->fd and MUST close it after use. */
uGDSError_t uGDSExportDmabuf(const void* bufPtr_base,
                              uGDSDmabufExport_t* out);

ssize_t uGDSRead(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                   off_t file_offset, off_t bufPtr_offset);

ssize_t uGDSWrite(uGDSHandle_t fh, const void* bufPtr_base, size_t size,
                    off_t file_offset, off_t bufPtr_offset);

/* ── Batch IO ── */

typedef void* uGDSBatchHandle_t;

typedef enum uGDSOpcode {
    UGDS_READ  = 0,
    UGDS_WRITE = 1,
} uGDSOpcode_t;

typedef enum uGDSBatchStatus {
    UGDS_BATCH_WAITING   = 0x01,
    UGDS_BATCH_PENDING   = 0x02,
    UGDS_BATCH_INVALID   = 0x04,
    UGDS_BATCH_COMPLETE  = 0x10,
    UGDS_BATCH_TIMEOUT   = 0x20,
    UGDS_BATCH_FAILED    = 0x40,
} uGDSBatchStatus_t;

typedef struct uGDSIOParams {
    void*           devPtr_base;
    off_t           file_offset;
    off_t           devPtr_offset;
    size_t          size;
    uGDSOpcode_t    opcode;
    void*           cookie;
} uGDSIOParams_t;

typedef struct uGDSIOEvents {
    void*               cookie;
    uGDSBatchStatus_t   status;
    ssize_t             ret;
} uGDSIOEvents_t;

uGDSError_t uGDSBatchIOSetUp(uGDSBatchHandle_t* batch, uGDSHandle_t fh,
                               unsigned nr);

uGDSError_t uGDSBatchIOSubmit(uGDSBatchHandle_t batch, unsigned nr,
                               uGDSIOParams_t* iocb, unsigned flags);

uGDSError_t uGDSBatchIOGetStatus(uGDSBatchHandle_t batch, unsigned min_nr,
                                  unsigned* nr, uGDSIOEvents_t* events,
                                  struct timespec* timeout);

void uGDSBatchIODestroy(uGDSBatchHandle_t batch);

/* ── Async Stream IO ──
 * Pointer params (size_p, file_offset_p, etc.) must be host-accessible.
 * Use cudaHostAlloc/hipHostMalloc for GPU-writable pinned memory (late binding).
 *
 * Stream parameter is void* to support both CUDA and HIP backends.
 * Pass cudaStream_t (CUDA) or hipStream_t (HIP) — both implicitly
 * convert to void*.
 *
 * Backend dispatch:
 *   - CUDA-only build: uses cudaLaunchHostFunc
 *   - HIP-only build: uses hipLaunchHostFunc
 *   - Dual-backend build: dispatches based on the buffer's registered
 *     backend (UGDS_BACKEND_CUDA → cudaLaunchHostFunc,
 *     UGDS_BACKEND_HIP → hipLaunchHostFunc). */

uGDSError_t uGDSReadAsync(uGDSHandle_t fh, void *bufPtr_base,
                           size_t *size_p, off_t *file_offset_p,
                           off_t *bufPtr_offset_p, ssize_t *bytes_read_p,
                           void* stream);

uGDSError_t uGDSWriteAsync(uGDSHandle_t fh, void *bufPtr_base,
                            size_t *size_p, off_t *file_offset_p,
                            off_t *bufPtr_offset_p, ssize_t *bytes_written_p,
                            void* stream);

uGDSError_t uGDSStreamRegister(void* stream);

uGDSError_t uGDSStreamDeregister(void* stream);

#ifdef __cplusplus
}
#endif

#endif /* __UGDS_H__ */
