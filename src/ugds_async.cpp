#include "ugds_internal.h"
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
/* Dual-backend: avoid both cuda_runtime.h and hip_runtime.h in this TU
 * to prevent type conflicts. Functions are declared extern "C" below. */
#define CUDART_CB
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__NVCC__)
#include <hip/hip_runtime.h>
#define CUDART_CB
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
typedef hipStream_t cudaStream_t;
#else
#include <cuda_runtime.h>
#endif
#include <mutex>
#include <cerrno>
#include <new>

#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(((size_t)1 << (sizeof(ssize_t) * 8 - 1)) - 1))
#endif

/* Forward declaration for dual-backend stream validation */
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
static uGDSError_t async_check_stream_backend(void* stream, const void* bufPtr_base);
#endif

/* Backend-neutral async IO.
 * Public API accepts void* for stream -- callers pass cudaStream_t
 * or hipStream_t, which implicitly convert. Internal dispatch
 * selects the correct backend launch function at compile time. */

static void async_io_callback(void* userData)
{
    AsyncRequest* req = static_cast<AsyncRequest*>(userData);
    size_t size = *req->size_p;
    off_t file_offset = *req->file_offset_p;
    off_t bufPtr_offset = *req->bufPtr_offset_p;

    ssize_t ret = do_io_internal(req->fh, req->bufPtr_base, size,
                                  file_offset, bufPtr_offset, req->opcode);
    *req->bytes_done_p = ret;

    /* Release the in-flight reference held by async_validate.
     * do_io_internal manages its own reference for registered buffers,
     * so this accounts for the enqueue-time increment only. */
    {
        std::lock_guard<std::mutex> drv_lock(g_driver.lock);
        auto it = g_driver.buf_registry.find(req->bufPtr_base);
        if (it != g_driver.buf_registry.end())
            it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }

    /* Release handle reference so HandleDeregister can proceed. */
    handle_release(static_cast<HandleState*>(req->fh));

    delete req;
}

static uGDSError_t async_validate(uGDSHandle_t fh, void* bufPtr_base,
                                   size_t* size_p, off_t* file_offset_p,
                                   off_t* bufPtr_offset_p, ssize_t* bytes_done_p,
                                   std::shared_ptr<HandleState>* hs_sp_out)
{
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (fh == nullptr || bufPtr_base == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (size_p == nullptr || file_offset_p == nullptr ||
        bufPtr_offset_p == nullptr || bytes_done_p == nullptr)
        return make_error(UGDS_INVALID_VALUE);

    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    auto it = g_driver.buf_registry.find(bufPtr_base);
    if (it == g_driver.buf_registry.end())
        return make_error(UGDS_INVALID_VALUE);

    /* Controller affinity check: validate against the submitting
     * handle's controller. */
    if (it->second.map_ctrl == nullptr)
        return make_error(UGDS_INVALID_VALUE);

    /* Hold in-flight reference from enqueue until callback completes.
     * This prevents uGDSBufDeregister from unmapping the buffer
     * while the async request is queued but not yet executed. */
    it->second.in_flight.fetch_add(1, std::memory_order_acq_rel);

    /* Also hold a handle reference so HandleDeregister cannot free
     * the HandleState (QPs, controller) while the async callback
     * is pending. Use handle_lookup_locked since we already hold
     * g_driver.lock from the buffer registry lookup above. */
    HandleState* hs = handle_lookup_locked(fh, hs_sp_out);
    if (!hs) {
        /* Roll back buffer in_flight ref on handle acquire failure */
        it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
        return make_error(UGDS_INVALID_VALUE);
    }

    /* Controller affinity check: controller at registration must match
     * the submitting handle's controller. */
    if (it->second.map_ctrl != hs->ctrl) {
        it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
        handle_release(hs);
        hs_sp_out->reset();
        return make_error(UGDS_INVALID_VALUE);
    }

    return UGDS_OK;
}

static AsyncRequest* make_async_request(uGDSHandle_t fh, void* bufPtr_base,
                                         size_t* size_p, off_t* file_offset_p,
                                         off_t* bufPtr_offset_p, ssize_t* bytes_done_p,
                                         uint8_t opcode,
                                         std::shared_ptr<HandleState> hs_sp)
{
    AsyncRequest* req = new (std::nothrow) AsyncRequest{
        fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p,
        bytes_done_p, opcode, std::move(hs_sp)
    };
    return req;
}

/* -- Backend-specific host function launch --
 * Uses ugsd_stream_t (void*) internally. In dual-backend builds,
 * dispatches based on the buffer's registered backend. */

#if defined(__HIP_PLATFORM_AMD__) && !defined(_CUDA)
/* HIP-only build */
#include <hip/hip_runtime_api.h>

static uGDSError_t async_launch_host_func(ugsd_stream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)opcode;
    hipError_t err = hipLaunchHostFunc((hipStream_t)stream,
                                       async_io_callback, req);
    if (err != hipSuccess) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
}

#elif defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
/* Dual-backend build: runtime dispatch based on buffer's backend.
 * Cannot include both cuda_runtime.h and hip_runtime_api.h in the same
 * TU due to vector type conflicts. Declare the runtime functions as
 * extern "C" -- both libcudart and libamdhip64 export them.
 *
 * NOTE: On ROCm, hipLaunchHostFunc can map to hipLaunchHostFunc_spt
 * when HIP_API_PER_THREAD_DEFAULT_STREAM is enabled. This declaration
 * matches the default (non-SPT) ABI. For SPT support, build HIP-only. */

/* Declare the runtime launch functions directly without their headers.
 * Both accept (stream_ptr, callback, user_data) and return int-like. */
extern "C" {
int cudaLaunchHostFunc(void* stream, void (*callback)(void*), void* userData);
int hipLaunchHostFunc(void* stream, void (*callback)(void*), void* userData);
}

static uGDSError_t async_launch_host_func(ugsd_stream_t stream,
                                           AsyncRequest* req, uint8_t opcode,
                                           uGDSBackend_t backend)
{
    (void)opcode;
    int err;
    if (backend == UGDS_BACKEND_HIP) {
        err = hipLaunchHostFunc(stream, async_io_callback, req);
    } else {
        err = cudaLaunchHostFunc(stream, async_io_callback, req);
    }
    if (err != 0) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = err;
        return e;
    }
    return UGDS_OK;
}

#elif defined(_CUDA) || defined(__CUDACC__)
/* CUDA-only build */
#include <cuda_runtime.h>

static uGDSError_t async_launch_host_func(ugsd_stream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)opcode;
    cudaError_t err = cudaLaunchHostFunc((cudaStream_t)(uintptr_t)stream,
                                         async_io_callback, req);
    if (err != cudaSuccess) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
}

#else
/* No GPU backend: async IO not available */
static uGDSError_t async_launch_host_func(ugsd_stream_t stream,
                                           AsyncRequest* req, uint8_t opcode)
{
    (void)stream; (void)opcode;
    delete req;
    return make_error(UGDS_IO_NOT_SUPPORTED);
}
#endif

static void async_release_inflight(void* bufPtr_base,
                                    std::shared_ptr<HandleState>* hs_sp)
{
    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    auto it = g_driver.buf_registry.find(bufPtr_base);
    if (it != g_driver.buf_registry.end())
        it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);

    if (hs_sp && *hs_sp) {
        handle_release(hs_sp->get());
        hs_sp->reset();
    }
}

/* -- Public async API (void* stream for dual-backend support) -- */

extern "C" uGDSError_t uGDSReadAsync(uGDSHandle_t fh, void* bufPtr_base,
                                       size_t* size_p, off_t* file_offset_p,
                                       off_t* bufPtr_offset_p, ssize_t* bytes_read_p,
                                       void* stream)
{
    std::shared_ptr<HandleState> hs_sp;
    uGDSError_t st = async_validate(fh, bufPtr_base, size_p, file_offset_p,
                                     bufPtr_offset_p, bytes_read_p, &hs_sp);
    if (st.err != UGDS_SUCCESS) return st;

#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    /* Dual-backend: check stream/buffer backend compatibility BEFORE
     * allocating request to avoid leak on mismatch. */
    {
        uGDSError_t sbe = async_check_stream_backend(stream, bufPtr_base);
        if (sbe.err != UGDS_SUCCESS) {
            async_release_inflight(bufPtr_base, &hs_sp);
            return sbe;
        }
    }
#endif

    /* Keep a copy of hs_sp so we can release on launch failure.
     * The request also holds a copy. */
    AsyncRequest* req = make_async_request(fh, bufPtr_base, size_p, file_offset_p,
                                            bufPtr_offset_p, bytes_read_p, NVM_IO_READ,
                                            hs_sp);
    if (req == nullptr) {
        async_release_inflight(bufPtr_base, &hs_sp);
        return make_error(UGDS_INTERNAL_ERROR);
    }
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    /* Dual-backend: look up buffer's registered backend */
    uGDSBackend_t backend = UGDS_BACKEND_DEFAULT;
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto it = g_driver.buf_registry.find(bufPtr_base);
        if (it != g_driver.buf_registry.end())
            backend = it->second.backend;
    }
    {
        uGDSError_t est = async_launch_host_func((ugsd_stream_t)stream, req, NVM_IO_READ, backend);
        if (est.err != UGDS_SUCCESS)
            async_release_inflight(bufPtr_base, &hs_sp);
        return est;
    }
#else
    {
        uGDSError_t est = async_launch_host_func((ugsd_stream_t)stream, req, NVM_IO_READ);
        if (est.err != UGDS_SUCCESS)
            async_release_inflight(bufPtr_base, &hs_sp);
        return est;
    }
#endif
}

extern "C" uGDSError_t uGDSWriteAsync(uGDSHandle_t fh, void* bufPtr_base,
                                        size_t* size_p, off_t* file_offset_p,
                                        off_t* bufPtr_offset_p, ssize_t* bytes_written_p,
                                        void* stream)
{
    std::shared_ptr<HandleState> hs_sp;
    uGDSError_t st = async_validate(fh, bufPtr_base, size_p, file_offset_p,
                                     bufPtr_offset_p, bytes_written_p, &hs_sp);
    if (st.err != UGDS_SUCCESS) return st;

#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    /* Dual-backend: check stream/buffer backend compatibility BEFORE
     * allocating request to avoid leak on mismatch. */
    {
        uGDSError_t sbe = async_check_stream_backend(stream, bufPtr_base);
        if (sbe.err != UGDS_SUCCESS) {
            async_release_inflight(bufPtr_base, &hs_sp);
            return sbe;
        }
    }
#endif

    AsyncRequest* req = make_async_request(fh, bufPtr_base, size_p, file_offset_p,
                                            bufPtr_offset_p, bytes_written_p, NVM_IO_WRITE,
                                            hs_sp);
    if (req == nullptr) {
        async_release_inflight(bufPtr_base, &hs_sp);
        return make_error(UGDS_INTERNAL_ERROR);
    }
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    /* Dual-backend: look up buffer's registered backend */
    uGDSBackend_t backend = UGDS_BACKEND_DEFAULT;
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto it = g_driver.buf_registry.find(bufPtr_base);
        if (it != g_driver.buf_registry.end())
            backend = it->second.backend;
    }
    {
        uGDSError_t est = async_launch_host_func((ugsd_stream_t)stream, req, NVM_IO_WRITE, backend);
        if (est.err != UGDS_SUCCESS)
            async_release_inflight(bufPtr_base, &hs_sp);
        return est;
    }
#else
    {
        uGDSError_t est = async_launch_host_func((ugsd_stream_t)stream, req, NVM_IO_WRITE);
        if (est.err != UGDS_SUCCESS)
            async_release_inflight(bufPtr_base, &hs_sp);
        return est;
    }
#endif
}

#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
/* Dual-backend: track stream->backend mapping for cross-backend detection.
 * Users register streams via uGDSStreamRegisterEx with an explicit backend.
 * Unregistered streams are treated as UGDS_BACKEND_DEFAULT (no check). */
#include <unordered_map>

static std::mutex                                   s_stream_map_lock;
static std::unordered_map<void*, uGDSBackend_t>     s_stream_backends;

extern "C" uGDSError_t uGDSStreamRegister(void* stream)
{
    /* In dual-backend mode, StreamRegister without a backend hint
     * just records the stream as default. Use uGDSStreamRegisterEx
     * to associate an explicit backend. */
    if (stream != nullptr) {
        std::lock_guard<std::mutex> g(s_stream_map_lock);
        s_stream_backends[stream] = UGDS_BACKEND_DEFAULT;
    }
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamRegisterEx(void* stream, uGDSBackend_t backend)
{
    if (stream == nullptr)
        return UGDS_OK;
    if (backend != UGDS_BACKEND_CUDA && backend != UGDS_BACKEND_HIP)
        return make_error(UGDS_INVALID_VALUE);
    std::lock_guard<std::mutex> g(s_stream_map_lock);
    s_stream_backends[stream] = backend;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamDeregister(void* stream)
{
    if (stream != nullptr) {
        std::lock_guard<std::mutex> g(s_stream_map_lock);
        s_stream_backends.erase(stream);
    }
    return UGDS_OK;
}

/* Validate that the stream's backend matches the buffer's backend.
 * Returns UGDS_OK if compatible, error otherwise. */
static uGDSError_t async_check_stream_backend(void* stream, const void* bufPtr_base)
{
    /* NULL stream means default stream -- always allowed */
    if (stream == nullptr)
        return UGDS_OK;

    /* Look up buffer backend */
    uGDSBackend_t buf_backend = UGDS_BACKEND_DEFAULT;
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto it = g_driver.buf_registry.find(const_cast<void*>(bufPtr_base));
        if (it != g_driver.buf_registry.end())
            buf_backend = it->second.backend;
    }

    /* Look up stream backend */
    uGDSBackend_t stream_backend = UGDS_BACKEND_DEFAULT;
    {
        std::lock_guard<std::mutex> g(s_stream_map_lock);
        auto it = s_stream_backends.find(stream);
        if (it != s_stream_backends.end())
            stream_backend = it->second;
    }

    /* If both are known (non-default), they must match */
    if (buf_backend != UGDS_BACKEND_DEFAULT &&
        stream_backend != UGDS_BACKEND_DEFAULT &&
        buf_backend != stream_backend)
        return make_error(UGDS_INVALID_VALUE);

    return UGDS_OK;
}
#else
/* Single-backend: stream/backend mismatch cannot occur at runtime,
 * but validate the backend enum for API consistency. */
#if defined(__HIP_PLATFORM_AMD__)
#define UGDS_ACTIVE_BACKEND UGDS_BACKEND_HIP
#else
#define UGDS_ACTIVE_BACKEND UGDS_BACKEND_CUDA
#endif

extern "C" uGDSError_t uGDSStreamRegister(void* stream)
{
    (void)stream;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamRegisterEx(void* stream, uGDSBackend_t backend)
{
    (void)stream;
    if (backend != UGDS_ACTIVE_BACKEND)
        return make_error(UGDS_INVALID_VALUE);
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamDeregister(void* stream)
{
    (void)stream;
    return UGDS_OK;
}
#undef UGDS_ACTIVE_BACKEND
#endif

/* ======================================================================== */
/* Vectored async (uGDSReadvAsync / uGDSWritevAsync)                       */
/* Async path + timeout ownership handling.                                */
/* ======================================================================== */

/* ---- Dual-backend stream-backend lookup helper ----
 * Reads the registered backend of a stream exactly once under
 * s_stream_map_lock.  Returns UGDS_BACKEND_DEFAULT for unregistered or
 * NULL streams.  Single-backend builds collapse to a compile-time active
 * backend.  This helper is separate from async_check_stream_backend so
 * that the async-vectored launch decision can read the stream backend
 * exactly once (atomic snapshot). */
static uGDSBackend_t iov_stream_backend_lookup(void* stream)
{
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    if (stream == nullptr)
        return UGDS_BACKEND_DEFAULT;
    std::lock_guard<std::mutex> g(s_stream_map_lock);
    auto it = s_stream_backends.find(stream);
    if (it == s_stream_backends.end())
        return UGDS_BACKEND_DEFAULT;
    return it->second;
#else
    (void)stream;
#if defined(__HIP_PLATFORM_AMD__)
    return UGDS_BACKEND_HIP;
#else
    return UGDS_BACKEND_CUDA;
#endif
#endif
}

/* ---- launch_backend decision ----
 * A is the buffer backend when the list is homogeneous (the single
 * backend that every segment shares).  Single-backend builds reduce to
 * A == active backend and the "mixed" branches become unreachable at
 * compile time (the other backend cannot register buffers). */
static uGDSError_t iov_compute_launch_backend(uGDSBackend_t list_a,
                                              bool         mixed,
                                              uGDSBackend_t stream_backend,
                                              uGDSBackend_t* out)
{
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    if (!mixed) {
        /* Homogeneous list: every segment is on backend A. */
        if (stream_backend == UGDS_BACKEND_DEFAULT) {
            /* NULL/unregistered/registered-DEFAULT: launch on A. */
            *out = list_a;
            return UGDS_OK;
        }
        if (stream_backend == list_a) {
            *out = list_a;
            return UGDS_OK;
        }
        /* Registered on a backend that is not A: scalar-mismatch parity. */
        return make_error(UGDS_INVALID_VALUE);
    }
    /* Mixed list: accepted iff the stream has an explicit registered
     * backend; launch on that backend. */
    if (stream_backend == UGDS_BACKEND_DEFAULT)
        return make_error(UGDS_INVALID_VALUE);
    *out = stream_backend;
    return UGDS_OK;
#else
    /* Single-backend: mixed is unrepresentable, the active backend is
     * the only one.  Any non-DEFAULT stream registration was already
     * validated against the active backend at StreamRegisterEx. */
    (void)stream_backend;
    (void)mixed;
    *out = list_a;
    return UGDS_OK;
#endif
}

/* ---- async_validate_v ----
 * Under a single g_driver.lock hold:
 *  - resolve every segment in the registry (identity-only: dma/base/
 *    registered_length/backend), controller affinity check;
 *  - acquire in-flight refs all-or-nothing into req->owner via
 *    acquire_identity_only (registered-only);
 *  - capture a list_backend snapshot for the launch decision;
 *  - take a handle-operation reference via handle_lookup_locked.
 * Geometry (offset/size) is NOT validated here -- late binding.
 *
 * On success, fills req->resolved (identity-only SegView[]), req->owner,
 * req->hs_sp, and *out_list_backend / *out_mixed.  On failure, owner is
 * untouched (no refs acquired) and the caller frees req->resolved.  The
 * handle-operation ref is released on any failure path after lookup
 * succeeds. */
static uGDSError_t async_validate_v(uGDSHandle_t fh,
                                    uGDSIoSegment_t* segs,
                                    unsigned nr_segs,
                                    SegView* resolved,
                                    SglRefOwner* owner,
                                    std::shared_ptr<HandleState>* hs_sp_out,
                                    uGDSBackend_t* out_list_backend,
                                    bool* out_mixed)
{
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (fh == nullptr || segs == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (nr_segs == 0 || nr_segs > UGDS_IOV_MAX)
        return make_error(UGDS_INVALID_VALUE);

    /* No per-segment value validation here (late binding).  Only base != NULL
     * is checked so that registry lookup is well-defined. */
    for (unsigned i = 0; i < nr_segs; ++i) {
        if (segs[i].base == nullptr)
            return make_error(UGDS_INVALID_VALUE);
    }

    /* Take a single g_driver.lock to do: handle lookup + identity-only
     * reference acquisition.  The handle ref is needed so that
     * handle_in_flight prevents Deregister from tearing down the QPs
     * while the request is queued.  owner->acquire_identity_only takes
     * the same lock internally; the lock is not held across the acquire
     * call. */
    HandleState* hs = nullptr;
    {
        std::lock_guard<std::mutex> g(g_driver.lock);
        hs = handle_lookup_locked(fh, hs_sp_out);
    }
    if (!hs)
        return make_error(UGDS_INVALID_VALUE);

    /* RAII: ensure handle_release runs even if acquire_identity_only
     * throws (e.g., from its internal mutex acquisition). */
    struct HandleReleaseGuard {
        HandleState* hs;
        std::shared_ptr<HandleState>* sp;
        bool armed;
        ~HandleReleaseGuard() {
            if (armed) {
                handle_release(hs);
                sp->reset();
            }
        }
    } hguard{hs, hs_sp_out, true};

    /* Controller affinity check + ref acquisition.  handle_lookup_locked
     * succeeded, so hs is valid for the lifetime of *hs_sp_out. */
    int rc = owner->acquire_identity_only(
        segs, static_cast<uint32_t>(nr_segs),
        hs->ctrl, resolved);
    if (rc != 0) {
        /* hguard releases handle ref */
        return make_error(UGDS_INVALID_VALUE);
    }

    /* Success: disarm the guard. Caller owns the handle ref now. */
    hguard.armed = false;

    /* Snapshot segment backends from resolved[] (read under no lock; the
     * registry entries are pinned by owner for the lifetime of this
     * request).  mixed == true iff backends span more than one value;
     * list_a is the first non-DEFAULT backend, or DEFAULT if all are
     * DEFAULT (homogeneous-DEFAULT). */
    uGDSBackend_t a = UGDS_BACKEND_DEFAULT;
    bool mixed = false;
    for (unsigned i = 0; i < nr_segs; ++i) {
        uGDSBackend_t b = resolved[i].backend;
        if (i == 0) {
            a = b;
        } else if (b != a) {
            mixed = true;
            /* Keep a as the first-segment backend; for a mixed list the
             * launch decision ignores a and uses the stream backend. */
        }
    }
    /* If a == DEFAULT the list is homogeneous-DEFAULT; treat A as the
     * build's active backend for the decision (parity with the scalar
     * buffer-backend dispatch). */
#if defined(__HIP_PLATFORM_AMD__) && !defined(_CUDA)
    if (a == UGDS_BACKEND_DEFAULT) a = UGDS_BACKEND_HIP;
#elif !defined(__HIP_PLATFORM_AMD__)
    if (a == UGDS_BACKEND_DEFAULT) a = UGDS_BACKEND_CUDA;
#else
    /* Dual build: leave DEFAULT; iov_compute_launch_backend treats the
     * single buffer-backend snapshot path as DEFAULT => launch on A.  In
     * dual builds, homogeneous-DEFAULT means all segments are DEFAULT --
     * that means none were registered via uGDSBufRegisterEx with an
     * explicit backend, so the buffer backend parity is the CUDA (A)
     * side; iov_compute_launch_backend uses list_a as-is. */
#endif

    *out_list_backend = a;
    *out_mixed = mixed;
    return UGDS_OK;
}

/* ---- async_iov_execute (callback body) ----
 * Potentially throwing: do_iov_engine acquires std::mutex and may throw.
 * The caller (async_iov_callback) is the noexcept exception boundary. */
static void async_iov_execute(AsyncRequest* req)
{
    /* 1. Late-bind offset/size from user_segs and file_offset_p.  These
     *    reads race with the producer only if the caller violates the
     *    binding contract (documented in include/ugds.h); under that
     *    contract the values are stable from enqueue until the callback
     *    runs.  No registry access is needed: identity is already
     *    snapshotted in resolved[]. */
    const off_t file_offset = *req->file_offset_p;

    /* 2. Complete resolved[] geometry and run the same per-segment value
     *    checks as sync do_readv_writev, but using the snapshotted
     *    registered_length.  This closes the late-binding window. */
    HandleState* hs = req->hs_sp.get();
    const size_t page_size  = hs->ctrl->page_size;
    const size_t block_size = hs->block_size;

    ssize_t err_ret = 0;
    uint64_t total_size = 0;

    for (unsigned i = 0; i < req->nr_segs; ++i) {
        const uGDSIoSegment_t& us = req->user_segs[i];
        SegView& sv = req->resolved[i];

        if (us.size == 0) { err_ret = -EINVAL; break; }
        if (us.offset < 0) { err_ret = -EINVAL; break; }
        if ((static_cast<size_t>(us.offset) % page_size) != 0) {
            err_ret = -EINVAL; break;
        }
        if ((us.size % block_size) != 0) { err_ret = -EINVAL; break; }

        /* Exact-length bound against the snapshotted registered_length. */
        const uint64_t off = static_cast<uint64_t>(us.offset);
        const uint64_t sz  = static_cast<uint64_t>(us.size);
        if (off > sv.registered_length ||
            sz > (sv.registered_length - off)) {
            err_ret = -EINVAL; break;
        }

        if (sz > static_cast<uint64_t>(SSIZE_MAX) - total_size) {
            err_ret = -EINVAL; break;
        }
        total_size += sz;

        sv.page_start = off / page_size;
        sv.size       = us.size;
    }

    if (err_ret == 0) {
        if (file_offset < 0 ||
            (static_cast<size_t>(file_offset) % block_size) != 0)
            err_ret = -EINVAL;
    }

    if (err_ret != 0) {
        /* Validation failure: release handle ref, write -errno, and
         * delete req.  owner destructor releases the in-flight refs
         * (none parked -- the engine never ran). */
        *req->bytes_done_p = err_ret;
        handle_release(hs);
        delete req;
        return;
    }

    /* 3. Engine dispatch.  Pass our owner; on timeout the engine
     *    move-assigns it into qp.timeout_refs and we must NOT release. */
    IovEngineResult result = do_iov_engine(hs, req->resolved,
                                           static_cast<uint32_t>(req->nr_segs),
                                           file_offset, req->opcode,
                                           &req->owner, nullptr);

    /* 4. Publish the result. */
    *req->bytes_done_p = result.ret;

    /* 5. Release the handle-operation reference taken at enqueue. */
    handle_release(hs);

    /* 6. delete req: the owner destructor releases the in-flight refs
     *    unless the engine parked them on timeout (in which case
     *    req->owner was moved-from and is empty). */
    delete req;
}

/* ---- async_iov_callback (exception boundary) ----
 * Declared noexcept-equivalent: any exception is caught and translated
 * to -EIO so nothing propagates into the CUDA/HIP runtime. */
static void async_iov_callback(void* userData) noexcept
{
    AsyncRequest* req = static_cast<AsyncRequest*>(userData);
    try {
        async_iov_execute(req);
    } catch (...) {
        /* Real catch boundary: async_iov_execute is NOT noexcept,
         * so exceptions from do_iov_engine or mutex acquisition
         * arrive here. Park the failure and free the request so
         * the caller's bytes_done_p is writable exactly once. */
        if (req->bytes_done_p != nullptr)
            *req->bytes_done_p = -EIO;
        if (req->hs_sp)
            handle_release(req->hs_sp.get());
        delete req;
    }
}

/* ---- vectored launch dispatch ----
 * Selects cudaLaunchHostFunc vs hipLaunchHostFunc in dual builds based
 * on the enqueue-time launch_backend decision.  Single-backend builds
 * use the active runtime. */
static uGDSError_t iov_launch_host_func(void* stream, AsyncRequest* req)
{
#if defined(_CUDA) && defined(__HIP_PLATFORM_AMD__)
    int err;
    if (req->launch_backend == UGDS_BACKEND_HIP) {
        err = hipLaunchHostFunc(stream, async_iov_callback, req);
    } else {
        err = cudaLaunchHostFunc(stream, async_iov_callback, req);
    }
    if (err != 0) {
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = err;
        return e;
    }
    return UGDS_OK;
#elif defined(__HIP_PLATFORM_AMD__)
    hipError_t err = hipLaunchHostFunc((hipStream_t)stream,
                                       async_iov_callback, req);
    if (err != hipSuccess) {
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
#else
    cudaError_t err = cudaLaunchHostFunc((cudaStream_t)(uintptr_t)stream,
                                         async_iov_callback, req);
    if (err != cudaSuccess) {
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }
    return UGDS_OK;
#endif
}

/* ---- common enqueue + launch for uGDSReadvAsync / uGDSWritevAsync ---- */
static uGDSError_t do_readv_writev_async(uGDSHandle_t fh,
                                         uGDSIoSegment_t* segs,
                                         unsigned nr_segs,
                                         off_t* file_offset_p,
                                         ssize_t* bytes_done_p,
                                         void* stream,
                                         uint8_t opcode)
{
    /* Malformed-call checks.  These do not need the registry and
     * cannot race with Deregister. */
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (fh == nullptr || segs == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (nr_segs == 0 || nr_segs > UGDS_IOV_MAX)
        return make_error(UGDS_INVALID_VALUE);
    if (file_offset_p == nullptr || bytes_done_p == nullptr)
        return make_error(UGDS_INVALID_VALUE);

    /* Pre-zero bytes_done_p so a late callback-failure still reports a
     * deterministic value to the caller. */
    *bytes_done_p = 0;

    /* Validate base pointers before any allocation. */
    for (unsigned i = 0; i < nr_segs; ++i) {
        if (segs[i].base == nullptr)
            return make_error(UGDS_INVALID_VALUE);
    }

    /* Allocate the request and its resolved[] storage BEFORE acquiring
     * any reference, so that allocation failure cannot strand an
     * in_flight or handle_in_flight counter.  resolved[] uses the
     * inline slot for nr <= 4.
     * unique_ptr owns req until successful launch; on any exception or
     * early return it is automatically freed. */
    auto req_guard = std::make_unique<AsyncRequest>();
    AsyncRequest* req = req_guard.get();
    req->fh            = fh;
    req->file_offset_p = file_offset_p;
    req->bytes_done_p  = bytes_done_p;
    req->opcode        = opcode;
    req->user_segs     = segs;
    req->nr_segs       = nr_segs;
    /* bufPtr_base/size_p/bufPtr_offset_p are unused for the vectored path. */

    if (nr_segs <= 4) {
        req->resolved = req->inline_resolved;
        /* resolved_owns_heap stays false (default). */
    } else {
        SegView* heap = new (std::nothrow) SegView[nr_segs];
        if (heap == nullptr)
            return make_error(UGDS_OUT_OF_MEMORY);
        req->resolved = heap;
        req->resolved_owns_heap = true;  /* destructor owns it */
    }

    /* async_validate_v acquires in-flight refs + handle ref, fills
     * resolved (identity-only), and snapshots list backend + mixed.
     * The handle ref guard ensures handle_release is called on
     * exception between validation and successful launch. */
    uGDSBackend_t list_backend = UGDS_BACKEND_DEFAULT;
    bool mixed = false;
    uGDSError_t st = async_validate_v(fh, segs, nr_segs, req->resolved,
                                      &req->owner, &req->hs_sp,
                                      &list_backend, &mixed);
    if (st.err != UGDS_SUCCESS) {
        return st;
    }

    /* RAII: release handle ref if we leave scope without launching. */
    struct HandleRefGuard {
        HandleState* hs;
        bool armed;
        ~HandleRefGuard() { if (armed && hs) handle_release(hs); }
    } hguard{req->hs_sp.get(), true};

    /* Compute launch_backend ONCE from the snapshot list_backend and
     * a single read of the stream backend. */
    uGDSBackend_t stream_backend = iov_stream_backend_lookup(stream);
    uGDSBackend_t launch_backend = UGDS_BACKEND_DEFAULT;
    uGDSError_t lberr = iov_compute_launch_backend(list_backend, mixed,
                                                    stream_backend,
                                                    &launch_backend);
    if (lberr.err != UGDS_SUCCESS) {
        req->owner.release();
        /* hguard releases handle ref */
        return lberr;
    }
    req->launch_backend = launch_backend;

    /* Launch the callback on the selected backend's stream.  On failure
     * the caller can retry, so we must roll back every reference we
     * acquired and free the request. */
    uGDSError_t lc = iov_launch_host_func(stream, req);
    if (lc.err != UGDS_SUCCESS) {
        req->owner.release();
        /* hguard releases handle ref */
        return lc;
    }

    /* Launch succeeded: disarm guards, release ownership to callback. */
    hguard.armed = false;
    (void)req_guard.release();
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSReadvAsync(uGDSHandle_t fh, uGDSIoSegment_t* segs,
                                        unsigned nr_segs, off_t* file_offset_p,
                                        ssize_t* bytes_read_p, void* stream)
{
    try {
        return do_readv_writev_async(fh, segs, nr_segs, file_offset_p,
                                     bytes_read_p, stream, NVM_IO_READ);
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}

extern "C" uGDSError_t uGDSWritevAsync(uGDSHandle_t fh, uGDSIoSegment_t* segs,
                                         unsigned nr_segs, off_t* file_offset_p,
                                         ssize_t* bytes_written_p, void* stream)
{
    try {
        return do_readv_writev_async(fh, segs, nr_segs, file_offset_p,
                                     bytes_written_p, stream, NVM_IO_WRITE);
    } catch (const std::bad_alloc&) {
        return make_error(UGDS_OUT_OF_MEMORY);
    } catch (...) {
        return make_error(UGDS_INTERNAL_ERROR);
    }
}
