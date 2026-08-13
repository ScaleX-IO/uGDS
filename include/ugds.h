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
/* GPU runtime headers are NOT included in the public header.
 * cuda_runtime.h and hip_runtime_api.h define conflicting types
 * (vector types, stream types) that cannot coexist in a single TU.
 *
 * The async API uses void* for streams. Callers pass cudaStream_t
 * (CUDA) or hipStream_t (HIP), which implicitly convert to void*.
 * Backend-specific runtime headers are included only in internal
 * source files that need them.
 *
 * Migration note: applications that previously relied on ugds.h to
 * transitively include cuda_runtime.h must now include it directly. */

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
    UGDS_RDMA_MR_STILL_ACTIVE        = UGDS_BASE_ERR + 41,
    UGDS_BUSY                        = UGDS_BASE_ERR + 42,
    UGDS_OUT_OF_MEMORY               = UGDS_BASE_ERR + 43,
    UGDS_BAD_FILE_DESCRIPTOR         = UGDS_BASE_ERR + 44,
    UGDS_DMABUF_INVALID_FD           = UGDS_BASE_ERR + 45,
    UGDS_DMABUF_NOT_P2P              = UGDS_BASE_ERR + 46,
    UGDS_FD_LIMIT_REACHED            = UGDS_BASE_ERR + 47,
    UGDS_INTERRUPTED                 = UGDS_BASE_ERR + 48,
    UGDS_DEVICE_LOST                 = UGDS_BASE_ERR + 49,
    UGDS_INVALID_USER_ADDRESS        = UGDS_BASE_ERR + 50,
    UGDS_ASYNC_QUEUE_FULL            = UGDS_BASE_ERR + 51,
    UGDS_PIN_LIMIT_EXCEEDED          = UGDS_BASE_ERR + 53,
    UGDS_TIMED_OUT                   = UGDS_BASE_ERR + 54,
    UGDS_STRUCT_VERSION_MISMATCH     = UGDS_BASE_ERR + 55,
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
    case UGDS_PLATFORM_NOT_SUPPORTED:      return "platform not supported";
    case UGDS_DEVICE_NOT_SUPPORTED:        return "device not supported";
    case UGDS_INVALID_FILE_TYPE:           return "unsupported file type";
    case UGDS_INVALID_VALUE:               return "invalid arguments";
    case UGDS_MEMORY_ALREADY_REGISTERED:   return "memory already registered";
    case UGDS_MEMORY_NOT_REGISTERED:       return "memory not registered";
    case UGDS_HANDLE_NOT_REGISTERED:       return "handle not registered";
    case UGDS_INTERNAL_ERROR:              return "internal error";
    case UGDS_GPU_MEMORY_PINNING_FAILED:   return "GPU memory pinning failed";
    case UGDS_BATCH_CAPACITY_EXCEEDED:     return "batch capacity exceeded";

    case UGDS_BUSY:                        return "resource busy, retry";
    case UGDS_RDMA_MR_STILL_ACTIVE:       return "RDMA MR still active";
    case UGDS_OUT_OF_MEMORY:               return "out of memory";
    case UGDS_BAD_FILE_DESCRIPTOR:         return "bad file descriptor";
    case UGDS_DMABUF_INVALID_FD:           return "invalid dma-buf file descriptor";
    case UGDS_DMABUF_NOT_P2P:              return "dma-buf is not PCIe peer BAR";
    case UGDS_FD_LIMIT_REACHED:            return "file descriptor limit reached";
    case UGDS_INTERRUPTED:                 return "operation interrupted";
    case UGDS_DEVICE_LOST:                 return "device lost";
    case UGDS_INVALID_USER_ADDRESS:        return "invalid user address";
    case UGDS_ASYNC_QUEUE_FULL:            return "async queue full";
    case UGDS_PIN_LIMIT_EXCEEDED:          return "pin limit exceeded";
    case UGDS_TIMED_OUT:                   return "operation timed out";
    case UGDS_STRUCT_VERSION_MISMATCH:     return "struct version mismatch";
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

/* Open the uGDS driver: discover the NVMe controller and initialize
 * the global driver state.  Must be called once before any other API.
 * Returns UGDS_OK or an error if the controller cannot be opened. */
uGDSError_t uGDSDriverOpen(void);

/* Close the driver and release controller resources.  All handles and
 * buffers should be deregistered first. */
uGDSError_t uGDSDriverClose(void);

/* Register a file/Windows handle obtained from the user as a uGDS
 * handle.  'descr' describes the underlying OS handle type (opaque fd
 * on Linux, opaque handle on Windows, userspace FS).  The returned
 * *fh is the opaque handle used by all IO entry points. */
uGDSError_t uGDSHandleRegister(uGDSHandle_t* fh, uGDSDescr_t* descr);

/* Return the usable NVMe namespace capacity in bytes for a registered handle. */
uGDSError_t uGDSGetDeviceCapacity(uGDSHandle_t fh,
                                  uint64_t* capacity_bytes);

/* Deregister a handle (non-blocking, equivalent to DeregisterEx with
 * timeout_sec == -1).  Blocks until all in-flight operations on this
 * handle have drained. */
void uGDSHandleDeregister(uGDSHandle_t fh);

/* Deregister a handle with a drain timeout.
 * Returns UGDS_OK on success, or UGDS_BUSY if the timeout expires
 * before all in-flight operations (including batch handles) complete.
 * timeout_sec == 0 means non-blocking check only.
 * timeout_sec == -1 means infinite wait (equivalent to uGDSHandleDeregister).
 * timeout_sec < -1 means after-reset teardown: release resources retained
 * by timed-out I/O, skip wedged/in-flight checks, and free all resources.
 *
 * Force contract (timeout_sec < -1): before calling this function the
 * caller MUST quiesce ALL host API calls and callbacks associated with
 * this handle -- not only NVMe commands, but every uGDSRead/uGDSWrite,
 * uGDSReadv/uGDSWritev, batch, and async operation that has passed
 * handle_lookup, plus every async callback already enqueued on a
 * stream. No such call may execute or resume concurrently with force
 * teardown. After force returns, the handle is fully invalid and no
 * API may use it. The caller must have already reset the controller. */
uGDSError_t uGDSHandleDeregisterEx(uGDSHandle_t fh, int timeout_sec);

/* Register a device/host buffer for direct IO.  'bufPtr_base' is the
 * exact base pointer of the allocation; 'length' is in bytes.  'flags'
 * may include UGDS_REGISTER_DMABUF to use the AMD HIP/dma-buf path.
 * The registration creates the dma mapping used by subsequent IO calls;
 * it must be released by uGDSBufDeregister. */
uGDSError_t uGDSBufRegister(const void* bufPtr_base, size_t length, int flags);

/* Flag for uGDSBufRegister: use AMD HIP/dma-buf path */
#define UGDS_REGISTER_DMABUF  NVM_MAP_DMABUF

/* Buffer registration flags (defined locally to avoid pulling in
 * libnvm/nvm_dma.h, which transitively includes C++ <atomic>) */
#define NVM_MAP_DMABUF      0x1
#define NVM_MAP_RDMA        0x2    /* Retain dmabuf fd for RDMA use */
#define NVM_MAP_FORCE_CUDA  0x4    /* Force CUDA path (skip auto-probe) */

/* Release a buffer registration previously created by uGDSBufRegister
 * or uGDSBufRegisterEx.  Blocks until all in-flight IO on this buffer
 * has drained; returns UGDS_BUSY if the drain cannot complete. */
uGDSError_t uGDSBufDeregister(const void* bufPtr_base);

/* Backend identifier for dual-backend dispatch. */
typedef enum uGDSBackend {
    UGDS_BACKEND_DEFAULT = 0,
    UGDS_BACKEND_CUDA    = 1,
    UGDS_BACKEND_HIP     = 2,
    UGDS_BACKEND_EXTERNAL = 3,
} uGDSBackend_t;

/* Extended buffer registration with backend selection and export flag */
typedef struct uGDSBufConfig {
    uGDSBackend_t   backend;
    bool            enable_export;
} uGDSBufConfig_t;

/* Register a device/host buffer with an explicit backend selection
 * (CUDA or HIP) and optional dma-buf export flag.  In dual-backend
 * builds the backend drives async launch dispatch; in single-backend
 * builds it is validated against the compiled-in backend. */
uGDSError_t uGDSBufRegisterEx(const void* bufPtr_base, size_t length,
                               const uGDSBufConfig_t* config);

/* dma-buf export handle for RDMA registration.
 * fd is dup()'d -- caller owns it and MUST close after use. */
typedef struct uGDSDmabufExport {
    int       fd;
    uint64_t  offset;
    size_t    length;
} uGDSDmabufExport_t;

/* Export a dma-buf file descriptor for a registered buffer so it can
 * be registered with an RDMA NIC via ibv_reg_dmabuf_mr().  'out->fd'
 * is dup()'d -- the caller owns it and MUST close() it after use.
 * Requires the buffer to have been registered with NVM_MAP_RDMA at
 * uGDSBufRegister time. */
uGDSError_t uGDSExportDmabuf(const void* bufPtr_base,
                              uGDSDmabufExport_t* out);

/* RDMA MR registration handle.
 *
 * Lifecycle: uGDSRDMARegister() creates an MR via ibv_reg_dmabuf_mr()
 * and records it. uGDSRDMAUnregister() calls ibv_dereg_mr() and removes
 * the record. The caller MUST ensure that:
 *   1. No new work requests are posted to this MR before unregistering.
 *   2. All outstanding completions for this MR have been reaped.
 * Deregistering an MR that still has in-flight WRs is undefined
 * behavior (NIC may still be accessing the memory). */
typedef struct uGDSRDMARegion {
    void*           mr;          /* ibv_mr* (opaque) */
    uint64_t        iova;        /* IOVA / buffer VA for post-recv */
    uint32_t        lkey;
    uint32_t        rkey;
    uint64_t        token;       /* internal generation token */
} uGDSRDMARegion_t;

uGDSError_t uGDSRDMARegister(const void* bufPtr_base,
                              void* pd, int access_flags,
                              uGDSRDMARegion_t* region);

uGDSError_t uGDSRDMAUnregister(const void* bufPtr_base,
                                uGDSRDMARegion_t* region);

/* --- External dma-buf registration ----------------------------------- */

/* Versioned parameter structure for external dma-buf registration.
 * All reserved fields must be zero. */
#define UGDS_DMABUF_REG_PARAMS_VERSION_1 1u
#define UGDS_DMABUF_REQUIRE_P2P          (1u << 0)

typedef struct uGDSDmabufRegParams {
    uint32_t struct_size;       /* sizeof(uGDSDmabufRegParams_t) */
    uint16_t version;           /* UGDS_DMABUF_REG_PARAMS_VERSION_1 */
    uint16_t reserved0;         /* must be zero */
    int32_t  dmabuf_fd;         /* application-owned dma-buf fd */
    uint32_t flags;             /* UGDS_DMABUF_* */
    uint64_t dmabuf_offset;     /* buffer VA - allocation base */
    uint64_t reserved[4];       /* must be zero */
} uGDSDmabufRegParams_t;

/* Register an externally exported dma-buf fd for direct NVMe P2P DMA.
 *
 * The caller exports GPU memory as a dma-buf fd (e.g. via Intel Level
 * Zero zeMemGetAllocProperties) and passes it here.  uGDS duplicates
 * the fd (F_DUPFD_CLOEXEC) so the caller may close the original fd
 * after this call returns.
 *
 * With UGDS_DMABUF_REQUIRE_P2P, the kernel verifies that every mapped
 * page falls within a single PCI peer device's memory BAR; failure
 * returns UGDS_DMABUF_NOT_P2P.
 *
 * The buffer is tagged UGDS_BACKEND_EXTERNAL: it has no implicit
 * stream launch backend.  Async stream I/O on an external buffer
 * requires a stream explicitly registered as CUDA or HIP.
 *
 * bufPtr_base is the GPU virtual address of the allocation base.
 * length is the exact byte count (not page-rounded). */
uGDSError_t uGDSBufRegisterDmabuf(const void* bufPtr_base, size_t length,
                                    const uGDSDmabufRegParams_t* params);

/* Capability query for dma-buf support.  Combines compile-time library
 * support with kernel-side V2/strict-P2P/pin-accounting capabilities.
 * Zero-initialize the struct before calling. */
typedef struct uGDSDmabufCaps {
    bool lib_dmabuf;            /* UGDS_HAVE_DMABUF compiled in */
    bool kmod_dmabuf_v2;        /* kernel supports V2 ioctl */
    bool kmod_require_p2p;      /* kernel supports UGDS_DMABUF_REQUIRE_P2P */
    bool kmod_pin_accounting;   /* kernel enforces pin limits */
} uGDSDmabufCaps_t;

uGDSError_t uGDSQueryDmabufSupport(uGDSDmabufCaps_t* caps);

/* Synchronous read from the namespace into a registered device buffer.
 * 'bufPtr_base' must have been registered via uGDSBufRegister/
 * uGDSBufRegisterEx (exact value); 'bufPtr_offset' is the byte offset
 * inside that registration.  'size' must be a multiple of the namespace
 * block size and bounded by the controller MDTS.  Returns the number of
 * bytes read, or -errno on failure. */
ssize_t uGDSRead(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                   off_t file_offset, off_t bufPtr_offset);

/* Synchronous write to the namespace from a registered device buffer.
 * Argument and return conventions match uGDSRead. */
ssize_t uGDSWrite(uGDSHandle_t fh, const void* bufPtr_base, size_t size,
                    off_t file_offset, off_t bufPtr_offset);

/* -- Batch IO -- */

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

/* Create a batch IO context bound to 'fh' with capacity for 'nr'
 * entries.  Allocates the batch's private submission/completion
 * resources and a dedicated QP (UGDS_BATCH_QUEUE_DEPTH).  The handle
 * is pinned for the lifetime of the batch via an internal shared_ptr.
 * Only one batch may be active per handle at a time; a second SetUp
 * returns UGDS_BUSY. */
uGDSError_t uGDSBatchIOSetUp(uGDSBatchHandle_t* batch, uGDSHandle_t fh,
                               unsigned nr);

/* Submit up to 'nr' plain (single-buffer) IO entries.  Each iocb[k]
 * describes a single devPtr_base/file_offset/size transfer.  Entries
 * are appended to the batch ring; completions are reaped via
 * uGDSBatchIOGetStatus.  Vectored entries may be mixed on the same
 * batch via uGDSBatchIOSubmitv. */
uGDSError_t uGDSBatchIOSubmit(uGDSBatchHandle_t batch, unsigned nr,
                               uGDSIOParams_t* iocb, unsigned flags);

/* Reap completions.  Blocks until at least 'min_nr' entries have
 * completed or 'timeout' elapses, then copies up to *nr events into
 * 'events'.  On return *nr holds the number actually reaped (0 on
 * timeout).  Events may be returned in any order. */
uGDSError_t uGDSBatchIOGetStatus(uGDSBatchHandle_t batch, unsigned min_nr,
                                  unsigned* nr, uGDSIOEvents_t* events,
                                  struct timespec* timeout);

/* Destroy the batch context.  Drains in-flight commands under a
 * bounded timeout.  If the drain fails the handle is wedged and the
 * batch transitions to WEDGED; the caller must reset the controller
 * and then call uGDSHandleDeregisterEx with timeout_sec < -1 to force
 * teardown. */
void uGDSBatchIODestroy(uGDSBatchHandle_t batch);

/* -- Vectored (scatter-gather) IO -- */

/* One scatter-gather segment. 'base' must be a pointer previously
 * registered via uGDSBufRegister/uGDSBufRegisterEx (exact value).
 * 'offset' is a byte offset inside that registration and must be a
 * multiple of the controller page size (MPS).  'size' must be a
 * multiple of the namespace block size and non-zero.  offset + size
 * must not exceed the length passed at registration (the exact byte
 * length, not a page-rounded value).
 *
 * The registration invariants enforced at acquire time are:
 *   - registered-range bound on offset + size
 *   - controller affinity between the buffer and the submitting handle */
typedef struct uGDSIoSegment {
    void*   base;     /* registered buffer base (exact registry key) */
    off_t   offset;   /* byte offset inside base; must be MPS-aligned */
    size_t  size;     /* transfer size in bytes; must be block-multiple */
} uGDSIoSegment_t;

/* Maximum number of segments per sync/async vectored call.
 * Mirrors POSIX IOV_MAX so callers can reuse existing iov sizing. */
#define UGDS_IOV_MAX        1024

/* Maximum number of segments per vectored batch entry.  Bounds the
 * arena allocation: capacity * UGDS_BATCH_IOV_MAX SegView slots are
 * reserved once per batch object lifetime. */
#define UGDS_BATCH_IOV_MAX  128

/* Vectored read.  Segments are consumed in array order: segs[0] maps
 * to [file_offset, file_offset + segs[0].size), segs[1] continues at
 * file_offset + segs[0].size, and so on.  Returns total bytes read, or
 * -errno (same convention as uGDSRead/uGDSWrite).
 *
 * The segment array is fully consumed during the call (synchronous);
 * the caller may free/reuse it on return.
 *
 * Validation is performed up-front under g_driver.lock (exact-length
 * bound, controller affinity check, MPS alignment of offset,
 * block-multiple of size, overflow-safe total).  On timeout the handle
 * is marked wedged and -EIO is returned; a controller reset is then
 * required before the handle can be reused. */
ssize_t uGDSReadv(uGDSHandle_t fh, const uGDSIoSegment_t* segs,
                    unsigned nr_segs, off_t file_offset);

/* Vectored write.  Layout and error conventions match uGDSReadv. */
ssize_t uGDSWritev(uGDSHandle_t fh, const uGDSIoSegment_t* segs,
                     unsigned nr_segs, off_t file_offset);

/* Vectored batch IO.  Because uGDSIOParams_t has no mode/union/reserved
 * field, SGL batch entries use this new params struct and submit function;
 * setup/status/destroy are shared with the existing batch object.
 *
 * 'segs' is copied at submit, so the caller may free iocb and the arrays
 * on return -- parity with uGDSBatchIOSubmit. */
typedef struct uGDSIOSegParams {
    const uGDSIoSegment_t* segs;      /* copied at submit */
    unsigned               nr_segs;   /* <= UGDS_BATCH_IOV_MAX */
    off_t                  file_offset;
    uGDSOpcode_t           opcode;
    void*                  cookie;
} uGDSIOSegParams_t;

/* Submit vectored (scatter-gather) batch entries.  Same semantics as
 * uGDSBatchIOSubmit: entries join the batch created by uGDSBatchIOSetUp;
 * completions are reaped via uGDSBatchIOGetStatus.  Plain and vectored
 * submits may be mixed on the same batch handle.
 *
 * Per-entry validation follows the value matrix (alignment, sizes,
 * overflow-safe total) and the analytic window counter.  Preflight
 * failures become terminal FAILED entries with -EINVAL and contribute
 * zero commands to the work array. */
uGDSError_t uGDSBatchIOSubmitv(uGDSBatchHandle_t batch, unsigned nr,
                                 uGDSIOSegParams_t* iocb, unsigned flags);

/* Vectored async read on a CUDA/HIP stream.
 * Binding contract (mirrors uGDSReadAsync late binding):
 *   - segs (the array pointer) and nr_segs are fixed at enqueue.
 *   - segs[i].base is read at enqueue (validation + in-flight refs)
 *     and MUST NOT change afterwards.
 *   - segs[i].offset, segs[i].size and *file_offset_p are read in the
 *     stream callback (late binding).  The segs array must remain
 *     valid until the callback runs.
 *
 * *bytes_read_p is pre-zeroed at enqueue. On validation or launch
 *     failure the return value remains zero and the error is returned
 *     via the uGDSError_t return code. On success the callback writes
 *     the byte count (or -errno on runtime failure) exactly once. */
uGDSError_t uGDSReadvAsync(uGDSHandle_t fh, uGDSIoSegment_t* segs,
                             unsigned nr_segs, off_t* file_offset_p,
                             ssize_t* bytes_read_p, void* stream);

/* Vectored async write on a CUDA/HIP stream.  Binding contract and
 * lifecycle match uGDSReadvAsync. */
uGDSError_t uGDSWritevAsync(uGDSHandle_t fh, uGDSIoSegment_t* segs,
                              unsigned nr_segs, off_t* file_offset_p,
                              ssize_t* bytes_written_p, void* stream);

/* -- Async Stream IO --
 * Pointer params (size_p, file_offset_p, etc.) must be host-accessible.
 * Use cudaHostAlloc/hipHostMalloc for GPU-writable pinned memory (late binding).
 *
 * Stream parameter is void* to support both CUDA and HIP backends.
 * Pass cudaStream_t (CUDA) or hipStream_t (HIP) -- both implicitly
 * convert to void*.
 *
 * Backend dispatch:
 *   - CUDA-only build: uses cudaLaunchHostFunc
 *   - HIP-only build: uses hipLaunchHostFunc
 *   - Dual-backend build: dispatches based on the buffer's registered
 *     backend (UGDS_BACKEND_CUDA -> cudaLaunchHostFunc,
 *     UGDS_BACKEND_HIP -> hipLaunchHostFunc). */

uGDSError_t uGDSReadAsync(uGDSHandle_t fh, void *bufPtr_base,
                           size_t *size_p, off_t *file_offset_p,
                           off_t *bufPtr_offset_p, ssize_t *bytes_read_p,
                           void* stream);

uGDSError_t uGDSWriteAsync(uGDSHandle_t fh, void *bufPtr_base,
                            size_t *size_p, off_t *file_offset_p,
                            off_t *bufPtr_offset_p, ssize_t *bytes_written_p,
                            void* stream);

/* Register a stream without a backend hint (treated as default).
 * In dual-backend builds, streams registered via this function are
 * not validated against the buffer's backend. Use uGDSStreamRegisterEx
 * to enable cross-backend mismatch detection. */
uGDSError_t uGDSStreamRegister(void* stream);

/* Register a stream with an explicit backend for dual-backend validation.
 * In dual-backend builds, uGDSReadAsync/uGDSWriteAsync will reject
 * stream/buffer backend mismatches when the stream has been registered
 * with an explicit backend. Streams registered via uGDSStreamRegister
 * or not registered at all bypass the check.
 *
 * In single-backend builds, this validates that the requested backend
 * matches the compiled-in backend (e.g. CUDA-only rejects HIP). */
uGDSError_t uGDSStreamRegisterEx(void* stream, uGDSBackend_t backend);

uGDSError_t uGDSStreamDeregister(void* stream);

#ifdef __cplusplus
}
#endif

#endif /* __UGDS_H__ */
