#ifndef __NVM_INTERNAL_DMA_H__
#define __NVM_INTERNAL_DMA_H__

#include <libnvm/nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>


/* Forward declaration */
struct va_range;


/*
 * dma-buf origin tagging.
 *
 * Replaces the former bool hip_origin with a typed enum so that
 * external (application-exported) dma-buf mappings can be
 * distinguished from HIP and CUDA exports.  The origin is used by
 * dual-backend dispatch and by uGDSBufRegister to pick the correct
 * launch backend.
 */
enum nvm_dmabuf_origin {
    NVM_DMABUF_ORIGIN_NONE     = 0,
    NVM_DMABUF_ORIGIN_HIP      = 1,
    NVM_DMABUF_ORIGIN_CUDA     = 2,
    NVM_DMABUF_ORIGIN_EXTERNAL = 3,
};


/*
 * Callback type for freeing an address range descriptor.
 * Called after the range is unmapped for the device and virtual address mapping can
 * be released.
 */
typedef void (*va_range_free_t)(struct va_range* va);



/*
 * Virtual address range descriptor.
 * This structure describes a custom address range mapped in userspace.
 */
struct va_range
{
    bool            remote;     // Indicates if this is remote memory
    volatile void*  vaddr;      // Virtual address of mapped address range
    size_t          page_size;  // Alignment of mapping (page size)
    size_t          n_pages;    // Number of pages for address range
};


#define VA_RANGE_INIT(remote, vaddr, page_size, n_pages)    \
    (struct va_range) {(remote), (vaddr), (page_size), (n_pages)}


/*
 * Map address range for a controller and create and initialize a DMA handle.
 */
int _nvm_dma_init(nvm_dma_t** handle,
                  const nvm_ctrl_t* ctrl,
                  struct va_range* va,
                  va_range_free_t release);



/*
 * Get the internal virtual address range from a handle.
 */
const struct va_range* _nvm_dma_va(const nvm_dma_t* handle);


/*
 * Set dmabuf metadata on a DMA handle's internal map.
 * Called by linux_dma.cpp AFTER the retain/close decision.
 */
int _nvm_dma_set_dmabuf_info(nvm_dma_t* handle,
                              int fd, uint64_t offset, size_t length);

/*
 * Set the dma-buf origin on a DMA handle's internal map.
 * Called after _nvm_dma_init for HIP, CUDA, or external mappings.
 */
void _nvm_dma_set_origin(nvm_dma_t* handle, enum nvm_dmabuf_origin origin);

/*
 * Retrieve the dma-buf origin of a DMA handle.
 * Used by dual-backend dispatch to select the correct async launch
 * and by uGDSBufRegister to choose the registration backend.
 */
enum nvm_dmabuf_origin nvm_dma_origin(const nvm_dma_t* handle);

/*
 * Retrieve dmabuf metadata from a DMA handle (internal only).
 * Returns 0 on success, -1 if handle is not dmabuf-backed.
 * Does NOT dup() -- returns internal fd. Callers must NOT close it.
 */
int nvm_dma_get_dmabuf_info(const nvm_dma_t* handle,
                             int* out_fd, uint64_t* out_offset, size_t* out_length);

/* --- Backward-compatibility inline shims -------------------------------
 *
 * Existing HIP code calls _nvm_dma_set_hip_origin / nvm_dma_is_hip_origin.
 * These inline shims forward to the new origin-based API so that
 * existing callers compile without modification. */

static inline void _nvm_dma_set_hip_origin(nvm_dma_t* handle)
{
    _nvm_dma_set_origin(handle, NVM_DMABUF_ORIGIN_HIP);
}

static inline bool nvm_dma_is_hip_origin(const nvm_dma_t* handle)
{
    return nvm_dma_origin(handle) == NVM_DMABUF_ORIGIN_HIP;
}

#endif /* __NVM_INTERNAL_DMA_H__ */
