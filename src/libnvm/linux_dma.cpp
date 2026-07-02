#ifndef __linux__
#error "Must compile for Linux"
#endif

#ifdef _CUDA
#ifndef __CUDA__
#define __CUDA__
#endif
#endif

#include <libnvm/nvm_types.h>
#include <libnvm/nvm_util.h>
#include <libnvm/nvm_dma.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include "internal/lib_util.h"
#include "internal/lib_ctrl.h"
#include "internal/dma.h"
#include "internal/map.h"
#include "internal/dprintf.h"

#ifdef _CUDA
#include <cuda.h>
#endif


static int posix_close_adapter(int fd)
{
    return close(fd);
}


static void remove_mapping_descriptor(struct ioctl_mapping* md)
{
    if (md->type == MAP_TYPE_API)
    {
        free((void*) md->buffer);
    }

    /* Close retained dmabuf fd exactly once via typed adapter */
    if (md->retain_fd && md->dmabuf_fd >= 0 && md->close_fn != NULL)
    {
        md->close_fn(md->dmabuf_fd);
        md->dmabuf_fd = -1;  /* prevent double-close */
    }

    free(md);
}



static void release_mapping_descriptor(struct va_range* va)
{
    remove_mapping_descriptor(_nvm_container_of(va, struct ioctl_mapping, range));
}



static int create_mapping_descriptor(struct ioctl_mapping** handle, size_t page_size, enum mapping_type type, void* buffer, size_t size)
{
    size_t n_pages = NVM_PAGE_ALIGN(size, page_size) / page_size;
    if (n_pages == 0)
    {
        return EINVAL;
    }

    struct ioctl_mapping* md = (struct ioctl_mapping*) malloc(sizeof(struct ioctl_mapping));
    if (md == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return errno;
    }

    md->type = type;
    md->buffer = buffer;
    md->range.remote = false;
    md->range.vaddr = (volatile void*) buffer;
    md->range.page_size = page_size;
    md->range.n_pages = n_pages;
    md->dmabuf_fd = -1;
    md->dmabuf_offset = 0;
    md->retain_fd = false;
    md->close_fn = NULL;

    *handle = md;
    return 0;
}



int nvm_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, size_t size)
{
    void* buffer;
    struct ioctl_mapping* md = NULL;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }

    *handle = NULL;
    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = posix_memalign(&buffer, ctrl->page_size, size);
    if (err != 0)
    {
        dprintf("Failed to allocate page-aligned memory buffer: %s\n", strerror(err));
        return err;
    }

    err = create_mapping_descriptor(&md, ctrl->page_size, MAP_TYPE_API, buffer, size);
    if (err != 0)
    {
        free(buffer);
        return err;
    }

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}



int nvm_dma_map_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t size)
{
    struct ioctl_mapping* md = NULL;
    *handle = NULL;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, ctrl->page_size, MAP_TYPE_HOST, vaddr, size);
    if (err != 0)
    {
        return err;
    }

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}



#ifdef _HIP
#include <unistd.h>
#include <fcntl.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime_api.h>

static int hsa_dmabuf_close_adapter(int fd)
{
    hsa_status_t s = hsa_amd_portable_close_dmabuf(fd);
    return (s == HSA_STATUS_SUCCESS) ? 0 : -1;
}

/* Use runtime page size -- must match kernel PAGE_SIZE for ioctl contract.
 * Thread-safe initialization via C++11 function-local static. */
static long hip_page_size()
{
    static const long ps = sysconf(_SC_PAGESIZE);
    return ps;
}

#define HPS (hip_page_size())

#endif

int nvm_dma_map_device_ex(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size, int flags)
{
    struct ioctl_mapping* md = NULL;
    int dmabuf_fd = -1;
    uint64_t dmabuf_offset = 0;
    int err;

    *handle = NULL;

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

#if !defined(_HIP) && !defined(_CUDA)
    dprintf("nvm_dma_map_device: no GPU backend compiled in\n");
    return ENOTSUP;
#endif

    /* Reject unknown flags */
    if (flags & ~(NVM_MAP_DMABUF | NVM_MAP_RDMA))
    {
        dprintf("nvm_dma_map_device: unknown flags 0x%x\n", flags);
        return EINVAL;
    }

    /* Determine which backend to use at runtime */
    int use_hip = 0;
#ifdef _HIP
  #ifdef _CUDA
    /* Dual-backend: explicit flag wins, otherwise probe pointer origin.
     * cuPointerGetAttribute succeeds for CUDA pointers; HIP pointers
     * fail it, so we fall back to HIP path. This handles on-the-fly
     * (unregistered) mappings where the caller doesn't know the backend.
     * Uses driver API to avoid CUDA/HIP runtime header conflicts. */
    if (flags & NVM_MAP_DMABUF)
    {
        use_hip = 1;
    }
    else if (!(flags & NVM_MAP_RDMA))
    {
        unsigned int mem_type = 0;
        CUresult cu_err = cuPointerGetAttribute(&mem_type,
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)devptr);
        if (cu_err != CUDA_SUCCESS)
        {
            /* Not a CUDA pointer — assume HIP */
            use_hip = 1;
        }
        else
        {
            use_hip = 0;
        }
    }
  #else
    use_hip = 1;  /* HIP-only build */
  #endif
#else
    /* CUDA-only or no-HIP build: reject dmabuf requests */
    if (flags & NVM_MAP_DMABUF)
    {
        dprintf("nvm_dma_map_device: NVM_MAP_DMABUF requested but HIP backend not compiled in\n");
        return ENOTSUP;
    }
#endif

#ifdef _HIP
    if (use_hip)
    {
        /* AMD HIP path: validate buffer, export as dma-buf */
        if (size == 0)
        {
            dprintf("nvm_dma_map_device: size must be non-zero\n");
            return EINVAL;
        }
        if (HPS <= 0)
        {
            dprintf("nvm_dma_map_device: sysconf(_SC_PAGESIZE) failed\n");
            return EINVAL;
        }

        /* Buffer validation (hipFile pattern) */
        hipPointerAttribute_t attrs;
        hipError_t hip_err = hipPointerGetAttributes(&attrs, devptr);
        if (hip_err != hipSuccess || attrs.type != hipMemoryTypeDevice)
        {
            dprintf("GPU pointer %p is not device memory\n", devptr);
            return EIO;
        }

        hipDeviceptr_t alloc_base;
        size_t alloc_size;
        hip_err = hipMemGetAddressRange(&alloc_base, &alloc_size, (hipDeviceptr_t)devptr);
        if (hip_err != hipSuccess)
        {
            dprintf("hipMemGetAddressRange failed for ptr %p\n", devptr);
            return EIO;
        }
        uintptr_t base_addr = (uintptr_t)alloc_base;
        uintptr_t ptr_addr = (uintptr_t)devptr;
        size_t offset_in_alloc = ptr_addr - base_addr;

        /* Compute page-aligned size with overflow check */
        if (size > SIZE_MAX - HPS)
        {
            dprintf("Buffer size %zu overflows alignment computation\n", size);
            return EINVAL;
        }
        size_t aligned_size = (size + HPS - 1) & ~(HPS - 1);

        /* Validate aligned size against allocation boundary */
        if (ptr_addr < base_addr || offset_in_alloc > alloc_size ||
            aligned_size > alloc_size - offset_in_alloc)
        {
            dprintf("Buffer range %p+%zu exceeds allocation %p+%zu\n",
                    devptr, size, (void*)base_addr, alloc_size);
            return EIO;
        }

        /* Validate page alignment */
        if (ptr_addr % HPS != 0)
        {
            dprintf("GPU pointer %p is not page-aligned\n", devptr);
            return EINVAL;
        }

        /* Export with PCIe P2P mapping type for NVMe-to-GPU peer DMA.
         * The v2 API with HSA_AMD_DMABUF_MAPPING_TYPE_PCIE is required for
         * correct NVMe<->GPU direct transfer -- it ensures the kernel receives
         * a PCIe BAR-mapped dmabuf suitable for peer-to-peer DMA.
         * HAVE_HSA_DMABUF_V2 is set by CMake check_cxx_source_compiles.
         * Without v2 support, the HIP backend cannot provide PCIe P2P. */
#ifndef HAVE_HSA_DMABUF_V2
        dprintf("HIP backend requires hsa_amd_portable_export_dmabuf_v2 support (ROCm >= 5.6). "
                "Please update ROCm or rebuild with CMake detection enabled.\n");
        return ENOTSUP;
#else
        hsa_status_t status = hsa_amd_portable_export_dmabuf_v2(
            devptr, aligned_size, &dmabuf_fd, &dmabuf_offset,
            HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
        if (status != HSA_STATUS_SUCCESS || dmabuf_fd < 0)
        {
            dprintf("hsa_amd_portable_export_dmabuf_v2() failed: %d\n", status);
            return EIO;
        }

        /* Set FD_CLOEXEC to prevent fd leak across exec.
         * If fcntl fails, close the fd and return error -- leaking
         * a dma-buf fd across exec is a security risk. */
        int fd_flags = fcntl(dmabuf_fd, F_GETFD);
        if (fd_flags < 0 || fcntl(dmabuf_fd, F_SETFD, fd_flags | FD_CLOEXEC) < 0)
        {
            dprintf("failed to set FD_CLOEXEC on dmabuf fd: %s\n", strerror(errno));
            hsa_amd_portable_close_dmabuf(dmabuf_fd);
            return EIO;
        }
#endif

        /* Validate offset is page-aligned */
        if (dmabuf_offset % HPS != 0)
        {
            dprintf("HSA dmabuf offset %llu is not page-aligned\n",
                    (unsigned long long)dmabuf_offset);
            hsa_amd_portable_close_dmabuf(dmabuf_fd);
            return EINVAL;
        }

        err = create_mapping_descriptor(&md, HPS, MAP_TYPE_DMABUF, devptr, size);
        if (err != 0)
        {
            hsa_amd_portable_close_dmabuf(dmabuf_fd);
            return err;
        }

        md->dmabuf_fd = dmabuf_fd;
        md->dmabuf_offset = dmabuf_offset;

        /* RDMA: retain fd for later export; set typed close adapter */
        if (flags & NVM_MAP_RDMA)
        {
            md->retain_fd = true;
            md->close_fn = hsa_dmabuf_close_adapter;
        }
    }
#endif /* _HIP */

#ifdef _CUDA
    if (!use_hip)
    {
        if (flags & NVM_MAP_RDMA)
        {
#ifdef HAVE_CUDA_DMABUF
            /* CUDA dma-buf export path for RDMA.
             * Full validation chain per approved design plan v8: */

            CUdevice cu_dev;
            int dma_buf_supported = 0;
            CUresult cu_err;

            /* Step 1: device attribute */
            cu_err = cuCtxGetDevice(&cu_dev);
            if (cu_err != CUDA_SUCCESS) {
                dprintf("cuCtxGetDevice failed: %d\n", cu_err);
                return EIO;
            }
            cu_err = cuDeviceGetAttribute(&dma_buf_supported,
                CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cu_dev);
            if (cu_err != CUDA_SUCCESS || !dma_buf_supported) {
                dprintf("CUDA device does not support dma-buf export\n");
                return ENOTSUP;
            }

            /* Step 2: pointer type validation */
            unsigned int mem_type = 0;
            cu_err = cuPointerGetAttribute(&mem_type,
                CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)devptr);
            if (cu_err != CUDA_SUCCESS) {
                dprintf("cuPointerGetAttribute(MEMORY_TYPE) failed: %d\n", cu_err);
                return EIO;
            }
            if (mem_type != CU_MEMORYTYPE_DEVICE) {
                dprintf("Pointer %p is not device memory (type=%u)\n", devptr, mem_type);
                return EINVAL;
            }

            /* Step 3: reject managed memory */
            int is_managed = 0;
            cu_err = cuPointerGetAttribute(&is_managed,
                CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)devptr);
            if (cu_err == CUDA_SUCCESS && is_managed) {
                dprintf("Managed memory not supported for GPUDirect RDMA\n");
                return EINVAL;
            }

            /* Step 4: allocation range + host page alignment */
            CUdeviceptr base_ptr;
            size_t alloc_size;
            cu_err = cuMemGetAddressRange(&base_ptr, &alloc_size,
                                          (CUdeviceptr)devptr);
            if (cu_err != CUDA_SUCCESS) {
                dprintf("cuMemGetAddressRange failed: %d\n", cu_err);
                return EIO;
            }

            long hps = sysconf(_SC_PAGESIZE);
            if (hps <= 0) {
                dprintf("sysconf(_SC_PAGESIZE) failed\n");
                return EINVAL;
            }
            /* Overflow guard: size + hps - 1 can wrap for huge sizes */
            if ((size_t)hps > 0 && size > SIZE_MAX - ((size_t)hps - 1)) {
                dprintf("Buffer size %zu overflows alignment computation\n", size);
                return EINVAL;
            }
            size_t offset_in_alloc = (uintptr_t)devptr - (uintptr_t)base_ptr;
            size_t aligned_size = (size + (size_t)hps - 1) & ~((size_t)hps - 1);

            if (offset_in_alloc + aligned_size > alloc_size) {
                dprintf("Buffer range %p+%zu exceeds CUDA allocation %p+%zu\n",
                        devptr, size, (void*)base_ptr, alloc_size);
                return EIO;
            }
            if ((uintptr_t)devptr % (size_t)hps != 0) {
                dprintf("CUDA pointer %p not host-page-aligned (%ld)\n", devptr, hps);
                return EINVAL;
            }

            /* Step 5: export with PCIe BAR mapping flag */
            dmabuf_fd = -1;
            cu_err = cuMemGetHandleForAddressRange(
                (void*)&dmabuf_fd,
                (CUdeviceptr)devptr,
                aligned_size,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);

            if (cu_err != CUDA_SUCCESS || dmabuf_fd < 0) {
                dprintf("cuMemGetHandleForAddressRange failed: %d\n", cu_err);
                return EIO;
            }
            dmabuf_offset = 0;

            /* Step 6: FD_CLOEXEC */
            int fd_flags = fcntl(dmabuf_fd, F_GETFD);
            if (fd_flags < 0 ||
                fcntl(dmabuf_fd, F_SETFD, fd_flags | FD_CLOEXEC) < 0) {
                dprintf("failed to set FD_CLOEXEC on CUDA dmabuf fd: %s\n",
                        strerror(errno));
                close(dmabuf_fd);
                return EIO;
            }

            /* Step 7: SYNC_MEMOPS — fatal for RDMA (BAR consistency) */
            int sync_val = 1;
            cu_err = cuPointerSetAttribute(&sync_val,
                CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)devptr);
            if (cu_err != CUDA_SUCCESS) {
                dprintf("cuPointerSetAttribute(SYNC_MEMOPS) failed: %d — "
                        "required for GPUDirect RDMA consistency\n", cu_err);
                close(dmabuf_fd);
                return EIO;
            }

            /* Create mapping with host PAGE_SIZE (not 64KiB) */
            err = create_mapping_descriptor(&md, (size_t)hps,
                                            MAP_TYPE_DMABUF_CUDA, devptr, size);
            if (err != 0) {
                close(dmabuf_fd);
                return err;
            }

            md->dmabuf_fd = dmabuf_fd;
            md->dmabuf_offset = 0;
            md->retain_fd = true;
            md->close_fn = posix_close_adapter;
#else
            dprintf("CUDA dma-buf export not compiled in (HAVE_CUDA_DMABUF undefined)\n");
            return ENOTSUP;
#endif
        }
        else
        {
            /* Standard NVIDIA CUDA path via kernel P2P */
            err = create_mapping_descriptor(&md, 1ULL << 16, MAP_TYPE_CUDA, devptr, size);
            if (err != 0)
            {
                return err;
            }
        }
    }
#endif /* _CUDA */

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        /* Ownership transferred to md. remove_mapping_descriptor will close
         * fd if retain_fd. Only close here if NOT retained (old behavior). */
        enum mapping_type saved_type = md->type;
        bool was_retained = md->retain_fd;
        remove_mapping_descriptor(md);
#ifdef _HIP
        if (saved_type == MAP_TYPE_DMABUF && !was_retained && dmabuf_fd >= 0)
            hsa_amd_portable_close_dmabuf(dmabuf_fd);
#endif
        return err;
    }

#ifdef _HIP
    if (md->type == MAP_TYPE_DMABUF)
    {
        /* Tag as HIP-origin for dual-backend async dispatch,
         * regardless of fd retention. */
        _nvm_dma_set_hip_origin(*handle);

        if (!md->retain_fd)
        {
            /* Non-RDMA: kernel holds refcount, close userspace fd */
            hsa_status_t close_status = hsa_amd_portable_close_dmabuf(dmabuf_fd);
            if (close_status != HSA_STATUS_SUCCESS) {
                dprintf("hsa_amd_portable_close_dmabuf() failed: %d\n", close_status);
                /* Unmap and release the handle we just initialized */
                nvm_dma_unmap(*handle);
                *handle = NULL;
                return EIO;
            }
            md->dmabuf_fd = -1;  /* cleared */
            _nvm_dma_set_dmabuf_info(*handle, -1, 0, 0);
        }
        else
        {
            /* RDMA: fd retained. Set map metadata with live fd. */
            _nvm_dma_set_dmabuf_info(*handle,
                dmabuf_fd, md->dmabuf_offset,
                md->range.page_size * md->range.n_pages);
        }
    }
#endif

#ifdef _CUDA
    /* CUDA dmabuf: fd is always retained (RDMA-only path).
     * Set map metadata with live fd. */
    if (md->type == MAP_TYPE_DMABUF_CUDA)
    {
        _nvm_dma_set_dmabuf_info(*handle,
            dmabuf_fd, 0,
            md->range.page_size * md->range.n_pages);
    }
#endif

    return 0;
}

/* Backward-compatible 4-arg wrapper (defaults to CUDA/flags=0) */
int nvm_dma_map_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size)
{
    return nvm_dma_map_device_ex(handle, ctrl, devptr, size, 0);
}
