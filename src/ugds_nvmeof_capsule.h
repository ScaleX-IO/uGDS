// User-space NVMe-oF/RDMA initiator whose data SGLs point at GPU memory.
//
// The kernel nvme-rdma path can only hand host pages to the target, so the
// default remote path stages data in host memory. This initiator opens its
// own NVMe-oF controller on the same subsystem, registers the caller's GPU
// allocation with the RDMA NIC through dma-buf, and puts that address and
// rkey in each command's keyed SGL. The target then RDMA-writes read data
// straight into GPU memory and RDMA-reads write data straight out of it.
//
// Built only with UGDS_ENABLE_NVMEOF_CAPSULE (needs librdmacm, libibverbs
// and the CUDA driver API).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>

struct ugds_capsule_ctrl;

struct ugds_capsule_target {
    std::string traddr;
    std::string trsvcid;
    std::string host_traddr;  // local source address; empty = routing table
    std::string subnqn;
    std::string hostnqn;
    uint8_t hostid[16] = {};
    uint32_t nsid = 1;
    int nr_io_queues = 8;
    int queue_depth = 16;  // clamped to the target's CAP.MQES + 1
};

// Describe the controller behind a kernel nvme-rdma namespace fd
// (/dev/nvmeXnY): same address, subsystem, namespace, host NQN and host ID.
// Returns 0 or a negative errno.
int ugds_capsule_target_from_fd(int fd, ugds_capsule_target* t);

// Opens an admin queue and t.nr_io_queues I/O queues, enables the controller,
// and starts a keep-alive thread. Returns 0 or a negative errno.
int ugds_capsule_connect(const ugds_capsule_target& t, ugds_capsule_ctrl** out);
void ugds_capsule_disconnect(ugds_capsule_ctrl* c);

uint64_t ugds_capsule_capacity(const ugds_capsule_ctrl* c);
uint32_t ugds_capsule_block_size(const ugds_capsule_ctrl* c);

// Registers the whole CUDA allocation that contains [gpu, gpu + size) with
// the NIC through dma-buf, once per allocation. Calls the CUDA driver API,
// so it must not run inside a CUDA host callback. I/O calls it on a miss.
int ugds_capsule_register(ugds_capsule_ctrl* c, const void* gpu, size_t size);

// Drops the NIC registration for the allocation containing gpu, if any.
void ugds_capsule_unregister(ugds_capsule_ctrl* c, const void* gpu);

struct ugds_capsule_op {
    void* gpu;
    size_t size;
    off_t off;
    bool is_write;
    ssize_t result;  // size on success, negative errno on failure
};

// Runs n ops on one I/O queue, keeping up to queue_depth commands in flight.
// gpu must be CUDA device memory; off and size must be multiples of the
// block size. Each op gets its own result. Returns 0 when the queue stayed
// usable, or a negative errno when the controller failed.
//
// Before each op the CUDA buffer ID of gpu is checked against the cached
// registration, so a freed and reallocated address is registered again
// instead of reaching stale memory. That check calls the CUDA driver API.
int ugds_capsule_submit(ugds_capsule_ctrl* c, ugds_capsule_op* ops, int n);

// One op; returns size or a negative errno.
ssize_t ugds_capsule_io(ugds_capsule_ctrl* c, void* gpu, size_t size, off_t off,
                        bool is_write);

int ugds_capsule_queue_depth(const ugds_capsule_ctrl* c);
int ugds_capsule_nr_queues(const ugds_capsule_ctrl* c);

// False once a transport error or command timeout stopped the controller.
// There is no reconnect; every queue is in the error state, so the target
// can no longer touch GPU memory through it.
bool ugds_capsule_alive(const ugds_capsule_ctrl* c);
