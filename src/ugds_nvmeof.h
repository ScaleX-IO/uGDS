// NVMe-oF remote path: kernel nvme-rdma block device + host bounce.
//
// Reached only through HandleState::remote_ep. The local P2P path does
// not include this header. Register does not start worker threads;
// ugds_nvmeof_io(..., on_lane=true) and the batch calls do, on first use.
// ugds_nvmeof_close() joins those threads. The caller still owns fd.
#pragma once
#include "ugds.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include <memory>

struct HandleState;
struct ugds_nvmeof_ep;

// BLKSSZGET + BLKGETSIZE64. Returns 0, or -errno / -EINVAL.
int ugds_nvmeof_probe(int fd, size_t* block_size_out, uint64_t* capacity_out);

// Open does not touch the device and does not create lanes. Close deletes
// the endpoint, joins lane threads if any were started, and disconnects the
// GPU-direct controller if one is attached.
ugds_nvmeof_ep* ugds_nvmeof_open(int fd, size_t block_size);
void ugds_nvmeof_close(ugds_nvmeof_ep* ep);

// Switch ep to GPU-direct data movement (UGDS_NVMEOF_GPU_DIRECT): open a
// user-space NVMe-oF controller on the subsystem behind ep's fd and check
// that it sees the same capacity and block size as the kernel. Afterwards
// ugds_nvmeof_io() and the batch calls never stage data in host memory.
// Returns 0, -ENOTSUP when the initiator is not built in, or another
// negative errno.
int ugds_nvmeof_attach_direct(ugds_nvmeof_ep* ep, uint64_t capacity);

// Reads or writes [buf + buf_offset, +size). file_offset and size must be
// multiples of the block size from probe().
//
// on_lane = false: copy + pread/pwrite on the calling thread (uGDSRead/Write).
// on_lane = true:  lane threads do the copy and the block I/O, and this call
//                  waits. Required from a CUDA host callback (uGDSReadAsync /
//                  WriteAsync), which is not allowed to call CUDA.
//
// Returns the byte count, or a negative errno.
ssize_t ugds_nvmeof_io(ugds_nvmeof_ep* ep, void* buf, size_t size,
                       off_t file_offset, off_t buf_offset, bool is_write,
                       bool on_lane);

// True when batch's first uint32_t is the remote tag ('RBN1'). Local
// BatchState's first field is a different tag ('LBN1'), so this is safe
// to call on either handle before any other field is read.
bool ugds_nvmeof_is_batch(const void* batch);
uGDSError_t ugds_nvmeof_batch_setup(void** batch_out, HandleState* hs,
                                    std::shared_ptr<HandleState> keep,
                                    unsigned nr);
uGDSError_t ugds_nvmeof_batch_submit(void* batch, unsigned nr,
                                     uGDSIOParams_t* iocb);
uGDSError_t ugds_nvmeof_batch_status(void* batch, unsigned min_nr,
                                     unsigned* nr, uGDSIOEvents_t* events,
                                     struct timespec* timeout);
void ugds_nvmeof_batch_destroy(void* batch);
