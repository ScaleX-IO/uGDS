#include "ugds_internal.h"
#include <mutex>

/* Backend-neutral async IO.
 * Public API accepts void* for stream — callers pass cudaStream_t
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
    {
        HandleState* hs = static_cast<HandleState*>(req->fh);
        hs->handle_in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }

    delete req;
}

static uGDSError_t async_validate(uGDSHandle_t fh, void* bufPtr_base,
                                   size_t* size_p, off_t* file_offset_p,
                                   off_t* bufPtr_offset_p, ssize_t* bytes_done_p)
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
     * is pending. */
    HandleState* hs = static_cast<HandleState*>(fh);
    hs->handle_in_flight.fetch_add(1, std::memory_order_acq_rel);

    return UGDS_OK;
}

static AsyncRequest* make_async_request(uGDSHandle_t fh, void* bufPtr_base,
                                         size_t* size_p, off_t* file_offset_p,
                                         off_t* bufPtr_offset_p, ssize_t* bytes_done_p,
                                         uint8_t opcode)
{
    AsyncRequest* req = new (std::nothrow) AsyncRequest{
        fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p,
        bytes_done_p, opcode
    };
    return req;
}

/* ── Backend-specific host function launch ──
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
 * extern "C" — both libcudart and libamdhip64 export them.
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

static void async_release_inflight(void* bufPtr_base, uGDSHandle_t fh)
{
    std::lock_guard<std::mutex> drv_lock(g_driver.lock);
    auto it = g_driver.buf_registry.find(bufPtr_base);
    if (it != g_driver.buf_registry.end())
        it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);

    HandleState* hs = static_cast<HandleState*>(fh);
    hs->handle_in_flight.fetch_sub(1, std::memory_order_acq_rel);
}

/* ── Public async API (void* stream for dual-backend support) ── */

extern "C" uGDSError_t uGDSReadAsync(uGDSHandle_t fh, void* bufPtr_base,
                                       size_t* size_p, off_t* file_offset_p,
                                       off_t* bufPtr_offset_p, ssize_t* bytes_read_p,
                                       void* stream)
{
    uGDSError_t st = async_validate(fh, bufPtr_base, size_p, file_offset_p,
                                     bufPtr_offset_p, bytes_read_p);
    if (st.err != UGDS_SUCCESS) return st;

    AsyncRequest* req = make_async_request(fh, bufPtr_base, size_p, file_offset_p,
                                            bufPtr_offset_p, bytes_read_p, NVM_IO_READ);
    if (req == nullptr) {
        async_release_inflight(bufPtr_base, fh);
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
        if (est.err != UGDS_SUCCESS) async_release_inflight(bufPtr_base, fh);
        return est;
    }
#else
    {
        uGDSError_t est = async_launch_host_func((ugsd_stream_t)stream, req, NVM_IO_READ);
        if (est.err != UGDS_SUCCESS) async_release_inflight(bufPtr_base, fh);
        return est;
    }
#endif
}

extern "C" uGDSError_t uGDSWriteAsync(uGDSHandle_t fh, void* bufPtr_base,
                                        size_t* size_p, off_t* file_offset_p,
                                        off_t* bufPtr_offset_p, ssize_t* bytes_written_p,
                                        void* stream)
{
    uGDSError_t st = async_validate(fh, bufPtr_base, size_p, file_offset_p,
                                     bufPtr_offset_p, bytes_written_p);
    if (st.err != UGDS_SUCCESS) return st;

    AsyncRequest* req = make_async_request(fh, bufPtr_base, size_p, file_offset_p,
                                            bufPtr_offset_p, bytes_written_p, NVM_IO_WRITE);
    if (req == nullptr) {
        async_release_inflight(bufPtr_base, fh);
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
        if (est.err != UGDS_SUCCESS) async_release_inflight(bufPtr_base, fh);
        return est;
    }
#else
    {
        uGDSError_t est = async_launch_host_func((ugsd_stream_t)stream, req, NVM_IO_WRITE);
        if (est.err != UGDS_SUCCESS) async_release_inflight(bufPtr_base, fh);
        return est;
    }
#endif
}

extern "C" uGDSError_t uGDSStreamRegister(void* stream)
{
    (void)stream;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamDeregister(void* stream)
{
    (void)stream;
    return UGDS_OK;
}
