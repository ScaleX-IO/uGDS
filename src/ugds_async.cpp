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
