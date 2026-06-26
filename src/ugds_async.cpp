#include "ugds_internal.h"
#include <cuda_runtime.h>
#include <mutex>

static void CUDART_CB async_io_callback(void* userData)
{
    AsyncRequest* req = static_cast<AsyncRequest*>(userData);
    size_t size = *req->size_p;
    off_t file_offset = *req->file_offset_p;
    off_t bufPtr_offset = *req->bufPtr_offset_p;

    ssize_t ret = do_io_internal(req->fh, req->bufPtr_base, size,
                                  file_offset, bufPtr_offset, req->opcode);
    *req->bytes_done_p = ret;
    delete req;
}

static uGDSError_t do_async(uGDSHandle_t fh, void* bufPtr_base,
                             size_t* size_p, off_t* file_offset_p,
                             off_t* bufPtr_offset_p, ssize_t* bytes_done_p,
                             cudaStream_t stream, uint8_t opcode)
{
    if (!g_driver.initialized)
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);
    if (fh == nullptr || bufPtr_base == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (size_p == nullptr || file_offset_p == nullptr ||
        bufPtr_offset_p == nullptr || bytes_done_p == nullptr)
        return make_error(UGDS_INVALID_VALUE);

    {
        std::lock_guard<std::mutex> drv_lock(g_driver.lock);
        if (g_driver.buf_registry.find(bufPtr_base) == g_driver.buf_registry.end())
            return make_error(UGDS_INVALID_VALUE);
    }

    AsyncRequest* req = new AsyncRequest{
        fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p,
        bytes_done_p, opcode
    };

    cudaError_t err = cudaLaunchHostFunc(stream, async_io_callback, req);
    if (err != cudaSuccess) {
        delete req;
        uGDSError_t e;
        e.err = UGDS_CUDA_DRIVER_ERROR;
        e.cu_err = static_cast<int>(err);
        return e;
    }

    return UGDS_OK;
}

extern "C" uGDSError_t uGDSReadAsync(uGDSHandle_t fh, void* bufPtr_base,
                                       size_t* size_p, off_t* file_offset_p,
                                       off_t* bufPtr_offset_p, ssize_t* bytes_read_p,
                                       cudaStream_t stream)
{
    return do_async(fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p,
                    bytes_read_p, stream, NVM_IO_READ);
}

extern "C" uGDSError_t uGDSWriteAsync(uGDSHandle_t fh, void* bufPtr_base,
                                        size_t* size_p, off_t* file_offset_p,
                                        off_t* bufPtr_offset_p, ssize_t* bytes_written_p,
                                        cudaStream_t stream)
{
    return do_async(fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p,
                    bytes_written_p, stream, NVM_IO_WRITE);
}

extern "C" uGDSError_t uGDSStreamRegister(cudaStream_t stream)
{
    (void)stream;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStreamDeregister(cudaStream_t stream)
{
    (void)stream;
    return UGDS_OK;
}
