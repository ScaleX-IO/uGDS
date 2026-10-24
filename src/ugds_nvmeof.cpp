// NVMe-oF remote path over the kernel nvme-rdma initiator.
//
// There is no GPU peer mapping to this device, so every byte is staged
// through a host bounce buffer and moved with pread/pwrite (or io_uring).
// The kernel selects the NVMe queue from the submitting CPU (blk-mq). The
// target only admits nr_tags commands on that queue (15 on the GP5000 we
// measured). One thread therefore fills only one queue.
//
// Two submission styles, chosen because they measured differently:
//
//  - uGDSRead/Write stays on the caller. A request of at most 16 MiB is a
//    single pread/pwrite, and the kernel plus the target split it. Cutting
//    one big read into lane-sized pieces was slower: extra bounces, and the
//    target's 256 KiB stripe (NOIOB) was crossed more often.
//  - BatchIO and Async need many commands in flight. Those go to "lanes":
//    one worker per hardware queue, pinned to a CPU that maps to that
//    queue, each with its own io_uring, pinned bounce slots, and CUDA
//    stream. Work is cut into <= 1 MiB chunks and round-robin'd across
//    lanes. 1 MiB matches nvme-rdma max_sectors_kb, so a chunk is already
//    one kernel bio.
//
// Lanes are not started by uGDSRead/Write. The first BatchIO or Async call
// builds them, and deregister joins them.
//
// GPU-direct mode (UGDS_NVMEOF_GPU_DIRECT, ugds_nvmeof_capsule.h) keeps the
// same entry points but drops the bounce: uGDSRead/Write call the user-space
// initiator on the caller's thread, and each lane submits up to queue_depth
// commands at a time through it. The kernel fd is then used to find the
// controller, to cross-check capacity, and as the fallback once the direct
// controller has stopped.
//
// Object lifetime, for one registered handle:
//
//   uGDSHandleRegisterNvmeof[Ex]
//     -> ugds_nvmeof_open()          endpoint: fd + block size, no threads
//     -> ugds_nvmeof_attach_direct() only with UGDS_NVMEOF_GPU_DIRECT
//   first BatchIO / Async
//     -> ugds_nvmeof_ep::start()     reads sysfs, spawns the lanes
//   uGDSHandleDeregister
//     -> existing in-flight drain    no I/O of this handle is running
//     -> ugds_nvmeof_close()         stop + join lanes, then disconnect the
//                                    direct controller, then free
//
// Threads: caller threads run uGDSRead/Write and the batch calls; each lane
// is one std::thread that only touches its own queue and slots; CUDA host
// functions (Async) block on a Waiter until lanes finish their chunks.
#include "ugds_internal.h"
#include "ugds_nvmeof.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <linux/fs.h>
#include <mutex>
#include <sched.h>
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__has_include) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define UGDS_NVMEOF_CUDA 1
#endif
#ifdef UGDS_HAVE_URING
#include <liburing.h>
#endif
#ifdef UGDS_HAVE_CAPSULE
#include "ugds_nvmeof_capsule.h"
#endif

struct ugds_capsule_ctrl;

namespace {

// Remote batches are a different object from local BatchState. Both are
// handed out as void*. The first uint32_t is a tag so Submit/GetStatus/
// Destroy can tell them apart without a side table. Local batches use
// 'LBN1' (see BatchState::tag); remote batches use 'RBN1'.
constexpr uint32_t kBatchTag = 0x52424E31u;  // 'RBN1'
// One lane submission. Equal to the nvme-rdma max_sectors_kb default, so
// the kernel does not split it again.
constexpr size_t kLaneChunk = 1u << 20;
// Largest bounce a caller-thread Read/Write will allocate. Bigger requests
// are several sequential pread/pwrite calls of this size, not one.
constexpr size_t kSyncPiece = 16u << 20;
// The kernel may expose one hardware queue per CPU (128 here). 16 lanes x 15
// tags is 240 commands in flight; fio on the same target already peaks at
// 16 jobs x QD16 (838K IOPS) with 64 x QD4 barely higher (850K), so more
// lanes would mostly add threads.
constexpr int kMaxLanes = 16;
// nr_tags is the real limit (15 on that target). This only rejects a
// runaway UGDS_NVMEOF_QD.
constexpr int kMaxDepth = 64;

// ---- host bounce ----------------------------------------------------------
// Caller threads copy from plain aligned memory on cudaStreamPerThread:
// pinning does not help a 4 KiB copy, and the legacy default stream
// serializes concurrent callers (16 threads: 117K vs 190K 4K IOPS).
// Lanes use pinned memory on their own stream; they copy up to 1 MiB.

// A host buffer that only grows. get(n) reuses the current allocation when
// it is large enough, otherwise frees it and allocates n bytes. Not thread
// safe: each owner (a thread_local on caller threads, a slot on a lane)
// uses its own. 4 KiB alignment satisfies O_DIRECT on every sector size the
// kernel reports.
struct Bounce {
    void* p = nullptr;
    size_t cap = 0;
    bool pinned;
    explicit Bounce(bool pinned_) : pinned(pinned_) {}
    ~Bounce() { release(); }
    Bounce(const Bounce&) = delete;
    Bounce& operator=(const Bounce&) = delete;

    void* get(size_t n) {
        if (n <= cap) return p;
        release();
#ifdef UGDS_NVMEOF_CUDA
        if (pinned) {
            if (cudaHostAlloc(&p, n, cudaHostAllocDefault) != cudaSuccess) p = nullptr;
        } else if (posix_memalign(&p, 4096, n) != 0) {
            p = nullptr;
        }
#else
        if (posix_memalign(&p, 4096, n) != 0) p = nullptr;
#endif
        cap = p ? n : 0;
        return p;
    }
    void release() {
#ifdef UGDS_NVMEOF_CUDA
        if (p && pinned) cudaFreeHost(p);
        else free(p);
#else
        free(p);
#endif
        p = nullptr;
        cap = 0;
    }
};

// Synchronous GPU<->host copy. stream == nullptr means cudaStreamPerThread
// (caller threads); lanes pass their private non-blocking stream. Returns 0
// or -EIO. Without CUDA headers (host-only test builds) it is a memcpy.
int copy(void* dst, const void* src, size_t n, bool to_gpu, void* stream) {
#ifdef UGDS_NVMEOF_CUDA
    auto kind = to_gpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : cudaStreamPerThread;
    if (cudaMemcpyAsync(dst, src, n, kind, s) != cudaSuccess) return -EIO;
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -EIO;
#else
    (void)to_gpu; (void)stream;
    memcpy(dst, src, n);
    return 0;
#endif
}

// pread/pwrite the full n bytes, retrying on EINTR and short transfers.
// A zero-byte transfer before n is reached (past end of device) is -EIO.
ssize_t block_io(int fd, void* host, size_t n, off_t off, bool is_write) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = is_write ? pwrite(fd, (char*)host + done, n - done, off + done)
                             : pread(fd, (char*)host + done, n - done, off + done);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return -errno;
        if (r == 0) return -EIO;
        done += (size_t)r;
    }
    return (ssize_t)n;
}

// ---- completion tracking --------------------------------------------------
//
// Waiter: one per waiting party. A single Async/on_lane call owns one on its
// stack; a RemoteBatch owns one for all of its Ops. `done` counts finished
// Ops and is guarded by m.
//
// Op: one user request (one uGDSRead on a lane, or one batch entry). `left`
// is the number of chunks still running; the Op and its Waiter must outlive
// every chunk, which is why callers wait for `finished` (or, for batches,
// Destroy waits for done == used) before letting them go out of scope.
//
// Req: one chunk of an Op as queued on a lane. Plain data, copied by value.

struct Waiter {
    std::mutex m;
    std::condition_variable cv;
    unsigned done = 0;
};

struct Op {
    std::atomic<int> left{0};
    std::atomic<int> err{0};
    Waiter* w = nullptr;
    // Batch event fields; guarded by w->m once left reaches 0.
    void* cookie = nullptr;
    size_t size = 0;
    bool finished = false;
    bool reported = false;
};

// One Op is one user request, possibly split into several lane chunks.
// The last chunk to finish publishes the op. The first error sticks;
// later chunks still run so the waiter is not left short one completion.
void chunk_done(Op* op, int err) {
    if (err) {
        int expect = 0;
        op->err.compare_exchange_strong(expect, err);
    }
    if (op->left.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    std::lock_guard<std::mutex> g(op->w->m);
    op->finished = true;
    op->w->done++;
    op->w->cv.notify_all();
}

struct Req {
    Op* op;
    void* gpu;
    off_t off;
    uint32_t len;
    bool is_write;
};

// ---- lanes ----------------------------------------------------------------
//
// One worker thread. Other threads only call push() (any thread) and set
// `stop` under m (the endpoint destructor). Everything else — slots, ring,
// stream, `busy` — is touched by the lane thread alone. `stop` makes the
// lane finish what is already queued and in flight, then exit; queued work
// is never dropped.

struct Lane {
    int fd = -1;
    int cpu = -1;
    int depth = 1;
    ugds_capsule_ctrl* direct = nullptr;  // set in GPU-direct mode
    std::mutex m;
    std::condition_variable cv;
    std::deque<Req> q;
    bool stop = false;
    std::thread thr;

    // Append chunks and wake the lane. Called from dispatch() on the
    // submitting thread.
    void push(std::vector<Req>& rs) {
        {
            std::lock_guard<std::mutex> g(m);
            q.insert(q.end(), rs.begin(), rs.end());
        }
        cv.notify_one();
    }

    // Take the oldest chunk. block = true waits for work or for stop; the
    // only way it returns false is stop set with the queue empty.
    // block = false never waits.
    bool pop(Req* r, bool block) {
        std::unique_lock<std::mutex> g(m);
        if (block) cv.wait(g, [&] { return stop || !q.empty(); });
        if (q.empty()) return false;
        *r = q.front();
        q.pop_front();
        return true;
    }

    // Bounce mode main loop, `depth` slots in flight. A slot is a pinned
    // buffer plus the Req using it (busy[s]); it is free again once its
    // completion has been copied out. The loop alternates between filling
    // free slots from the queue (blocking only when nothing is in flight)
    // and reaping io_uring completions.
    //
    // A write is copied GPU->host before the pread/pwrite is posted; a read
    // is copied host->GPU after the block I/O completes, on this lane's
    // private non-blocking stream.
    // Blocking here is fine: the caller is waiting on Waiter, and a CUDA
    // host callback is not allowed to call CUDA itself.
    //
    // Without liburing the same loop runs the block I/O inline and never
    // takes a slot, so depth does not add concurrency on that build.
    void run() {
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            sched_setaffinity(0, sizeof(set), &set);
        }
#ifdef UGDS_HAVE_CAPSULE
        if (direct) {
            run_direct();
            return;
        }
#endif
        void* stream = nullptr;
#ifdef UGDS_NVMEOF_CUDA
        cudaStream_t s;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) stream = s;
#endif
        std::vector<std::unique_ptr<Bounce>> slot;
        for (int i = 0; i < depth; ++i) slot.emplace_back(new Bounce(true));
        std::vector<Req> busy(depth);
        std::vector<int> free_slots;
        for (int i = depth - 1; i >= 0; --i) free_slots.push_back(i);
#ifdef UGDS_HAVE_URING
        io_uring ring;
        bool ring_ok = io_uring_queue_init(depth, &ring, 0) == 0;
#else
        bool ring_ok = false;
#endif
        int inflight = 0;
        for (;;) {
            Req r;
            while (!free_slots.empty() && pop(&r, inflight == 0)) {
                int s = free_slots.back();
                void* host = slot[s]->get(r.len);
                int e = host ? 0 : -ENOMEM;
                if (!e && r.is_write) e = copy(host, r.gpu, r.len, false, stream);
                if (!e && !ring_ok) {
                    ssize_t n = block_io(fd, host, r.len, r.off, r.is_write);
                    e = n < 0 ? (int)n : 0;
                    if (!e && !r.is_write) e = copy(r.gpu, host, r.len, true, stream);
                    chunk_done(r.op, e);
                    continue;
                }
                if (e) {
                    chunk_done(r.op, e);
                    continue;
                }
#ifdef UGDS_HAVE_URING
                free_slots.pop_back();
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (r.is_write) io_uring_prep_write(sqe, fd, host, r.len, r.off);
                else io_uring_prep_read(sqe, fd, host, r.len, r.off);
                io_uring_sqe_set_data(sqe, (void*)(uintptr_t)s);
                busy[s] = r;
                inflight++;
#endif
            }
            if (inflight == 0) {
                std::lock_guard<std::mutex> g(m);
                if (stop && q.empty()) break;
                continue;
            }
#ifdef UGDS_HAVE_URING
            io_uring_submit_and_wait(&ring, 1);
            io_uring_cqe* cqe;
            while (io_uring_peek_cqe(&ring, &cqe) == 0) {
                int s = (int)(uintptr_t)io_uring_cqe_get_data(cqe);
                int res = cqe->res;
                io_uring_cqe_seen(&ring, cqe);
                Req& d = busy[s];
                int e = res < 0 ? res : ((uint32_t)res != d.len ? -EIO : 0);
                if (!e && !d.is_write) e = copy(d.gpu, slot[s]->p, d.len, true, stream);
                chunk_done(d.op, e);
                free_slots.push_back(s);
                inflight--;
            }
#endif
        }
#ifdef UGDS_HAVE_URING
        if (ring_ok) io_uring_queue_exit(&ring);
#endif
        slot.clear();
#ifdef UGDS_NVMEOF_CUDA
        if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
#endif
    }

#ifdef UGDS_HAVE_CAPSULE
    // GPU-direct: no bounce, no io_uring, no CUDA stream. Wait for one
    // chunk, take whatever else is queued up to depth, and hand them to the
    // initiator as one submission. The target moves the bytes.
    //
    // Once the direct controller has stopped, chunks it failed and every
    // later chunk go through a host bounce and the kernel fd instead.
    void run_direct() {
        std::vector<Req> batch;
        std::vector<ugds_capsule_op> ops;
        Bounce fallback(true);
        for (;;) {
            Req r;
            if (!pop(&r, true)) break;  // stop requested and queue empty
            batch.assign(1, r);
            while ((int)batch.size() < depth && pop(&r, false)) batch.push_back(r);
            ops.resize(batch.size());
            for (size_t i = 0; i < batch.size(); ++i)
                ops[i] = {batch[i].gpu, batch[i].len, batch[i].off, batch[i].is_write, -EIO};
            if (ugds_capsule_alive(direct)) ugds_capsule_submit(direct, ops.data(), (int)ops.size());
            bool alive = ugds_capsule_alive(direct);
            for (size_t i = 0; i < batch.size(); ++i) {
                int e = ops[i].result < 0 ? (int)ops[i].result : 0;
                if (e && !alive) e = bounce_io(batch[i], fallback);
                chunk_done(batch[i].op, e);
            }
        }
    }

    // One chunk through a host bounce on the kernel fd, synchronously. Only
    // used after the direct controller has stopped.
    int bounce_io(const Req& r, Bounce& b) {
        void* host = b.get(r.len);
        if (!host) return -ENOMEM;
        if (r.is_write && copy(host, r.gpu, r.len, false, nullptr)) return -EIO;
        ssize_t n = block_io(fd, host, r.len, r.off, r.is_write);
        if (n < 0) return (int)n;
        if (!r.is_write && copy(r.gpu, host, r.len, true, nullptr)) return -EIO;
        return 0;
    }
#endif
};

// ---- topology: one lane per hardware queue --------------------------------

// "0-3,8,10-11" -> {0,1,2,3,8,10,11}, the sysfs cpulist format.
std::vector<int> parse_cpus(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e;
        long a = strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        long b = a;
        if (*e == '-') { p = e + 1; b = strtol(p, &e, 10); }
        for (long c = a; c <= b; ++c) v.push_back((int)c);
        p = e;
    }
    return v;
}

// First line of a sysfs file, or "" if it cannot be read.
std::string read_line(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[4096] = {};
    if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
    fclose(f);
    return buf;
}

// Sorted entries of a directory, without "." names; empty if missing.
std::vector<std::string> list_dir(const std::string& path) {
    std::vector<std::string> v;
    if (DIR* d = opendir(path.c_str())) {
        while (dirent* e = readdir(d))
            if (e->d_name[0] != '.') v.push_back(e->d_name);
        closedir(d);
    }
    std::sort(v.begin(), v.end());
    return v;
}

// /sys/block of the device the kernel actually queues on: the multipath
// head (nvmeXnY) has no mq/ directory, its path device (nvmeXcZnY) does.
std::string queue_dir(int fd) {
    char link[256] = {};
    std::string self = "/proc/self/fd/" + std::to_string(fd);
    if (readlink(self.c_str(), link, sizeof(link) - 1) <= 0) return "";
    std::string blk = "/sys/block/" + std::string(strrchr(link, '/') + 1);
    if (!list_dir(blk + "/mq").empty()) return blk;
    for (auto& p : list_dir(blk + "/multipath"))
        if (!list_dir("/sys/block/" + p + "/mq").empty()) return "/sys/block/" + p;
    return "";
}

}  // namespace

// Per-handle remote state, owned by HandleState::remote_ep.
//   fd, block_size  set at open, read-only afterwards; fd belongs to the caller
//   direct          set once by attach_direct before the handle is published
//   lanes           built once by start() under start_m, read-only afterwards
//   rr              round-robin cursor for dispatch(), any thread
struct ugds_nvmeof_ep {
    int fd;
    size_t block_size;
    ugds_capsule_ctrl* direct = nullptr;  // GPU-direct initiator, if attached
    std::mutex start_m;
    bool started = false;
    std::vector<std::unique_ptr<Lane>> lanes;
    std::atomic<uint32_t> rr{0};

    bool aligned(off_t off, size_t size, off_t buf_off) const {
        return off >= 0 && buf_off >= 0 && size > 0 &&
               (size_t)off % block_size == 0 && size % block_size == 0;
    }

    // Build the lanes on first use. Idempotent and thread safe; a second
    // caller waits on start_m and sees the result of the first. Returns
    // false only if no lane could be created.
    //
    // CPU choice: the kernel queue map (path device mq/<n>/cpu_list) decides
    // which CPU submits to which hardware queue. Taking one CPU per queue
    // spreads lanes over distinct queues; queues reachable from the
    // device's NUMA node are taken first.
    bool start() {
        std::lock_guard<std::mutex> g(start_m);
        if (started) return !lanes.empty();
        started = true;
        std::string qd = queue_dir(fd);
        std::string node = read_line(qd + "/device/numa_node");
        std::vector<int> near = parse_cpus(read_line(
            "/sys/devices/system/node/node" + (node.empty() ? "0" : node) + "/cpulist"));
        if (const char* env = getenv("UGDS_NVMEOF_CPUSET")) near = parse_cpus(env);
        int want = kMaxLanes;
        if (const char* env = getenv("UGDS_NVMEOF_LANES")) want = std::max(1, atoi(env));
        int direct_depth = 0;
#ifdef UGDS_HAVE_CAPSULE
        // One lane per initiator queue. The kernel queue map below still
        // supplies NUMA-local CPUs to pin them on.
        if (direct) {
            want = std::min(want, ugds_capsule_nr_queues(direct));
            direct_depth = ugds_capsule_queue_depth(direct);
        }
#endif

        // One CPU per hardware queue: covering every queue matters more than
        // NUMA locality, because each queue only admits nr_tags commands.
        // Queues that have a CPU on the device's node come first.
        std::vector<std::pair<int, int>> local, remote;  // {cpu, depth}
        for (auto& h : list_dir(qd + "/mq")) {
            std::vector<int> cpus = parse_cpus(read_line(qd + "/mq/" + h + "/cpu_list"));
            int tags = atoi(read_line(qd + "/mq/" + h + "/nr_tags").c_str());
            if (cpus.empty()) continue;
            auto it = std::find_first_of(cpus.begin(), cpus.end(), near.begin(), near.end());
            if (it != cpus.end()) local.push_back({*it, tags});
            else remote.push_back({cpus.front(), tags});
        }
        std::vector<std::pair<int, int>> pick = local;
        pick.insert(pick.end(), remote.begin(), remote.end());
        // No mq/ directory (older kernel, or the fd is not a block device
        // we can map). Still start workers so Batch/Async function; they
        // just share whatever CPUs the device's NUMA node lists.
        if (pick.empty())
            for (int c : near) pick.push_back({c, 16});
        if (pick.empty()) pick.push_back({-1, 16});
        pick.resize(std::min<size_t>(pick.size(), (size_t)want));

        int depth_env = 0;
        if (const char* env = getenv("UGDS_NVMEOF_QD")) depth_env = atoi(env);
        for (auto& [cpu, tags] : pick) {
            auto l = std::make_unique<Lane>();
            l->fd = fd;
            l->cpu = cpu;
            l->direct = direct;
            l->depth = direct ? direct_depth
                              : std::clamp(depth_env > 0 ? depth_env : tags, 1, kMaxDepth);
            l->thr = std::thread(&Lane::run, l.get());
            lanes.push_back(std::move(l));
        }
        return true;
    }

    // Cut [gpu, gpu+size) into kLaneChunk pieces. Round-robin is by
    // chunk, not by call, so one large request occupies every lane and
    // a following small request does not all land on lane 0.
    void dispatch(Op* op, void* gpu, size_t size, off_t off, bool is_write) {
        size_t n = (size + kLaneChunk - 1) / kLaneChunk;
        op->left.store((int)n, std::memory_order_relaxed);
        std::vector<std::vector<Req>> per(lanes.size());
        for (size_t i = 0; i < n; ++i) {
            size_t o = i * kLaneChunk;
            Req r{op, (char*)gpu + o, off + (off_t)o,
                  (uint32_t)std::min(kLaneChunk, size - o), is_write};
            per[rr.fetch_add(1, std::memory_order_relaxed) % lanes.size()].push_back(r);
        }
        for (size_t i = 0; i < lanes.size(); ++i)
            if (!per[i].empty()) lanes[i]->push(per[i]);
    }

    // Order matters: lanes are joined first because a direct-mode lane may
    // still be inside ugds_capsule_submit(); only then is the controller
    // disconnected. Deregister has already drained the handle, so the
    // queues are normally empty here.
    ~ugds_nvmeof_ep() {
        for (auto& l : lanes) {
            { std::lock_guard<std::mutex> g(l->m); l->stop = true; }
            l->cv.notify_one();
        }
        for (auto& l : lanes) l->thr.join();
#ifdef UGDS_HAVE_CAPSULE
        if (direct) ugds_capsule_disconnect(direct);
#endif
    }
};

int ugds_nvmeof_probe(int fd, size_t* block_size_out, uint64_t* capacity_out) {
    unsigned int ss = 0;
    unsigned long long bytes = 0;
    if (ioctl(fd, BLKSSZGET, &ss) != 0) return -errno;
    if (ioctl(fd, BLKGETSIZE64, &bytes) != 0) return -errno;
    if (ss == 0 || bytes == 0) return -EINVAL;
    *block_size_out = ss;
    *capacity_out = bytes;
    return 0;
}

ugds_nvmeof_ep* ugds_nvmeof_open(int fd, size_t block_size) {
    auto* ep = new (std::nothrow) ugds_nvmeof_ep();
    if (ep) {
        ep->fd = fd;
        ep->block_size = block_size;
    }
    return ep;
}

void ugds_nvmeof_close(ugds_nvmeof_ep* ep) { delete ep; }

// Opens the second (user-space) controller on the same subsystem as the
// kernel one. Called during register, before the handle is visible to other
// threads, so ep->direct needs no locking. The capacity / block-size check
// guards against the sysfs lookup having found a different namespace.
int ugds_nvmeof_attach_direct(ugds_nvmeof_ep* ep, uint64_t capacity) {
#ifdef UGDS_HAVE_CAPSULE
    ugds_capsule_target t;
    if (int r = ugds_capsule_target_from_fd(ep->fd, &t)) return r;
    t.nr_io_queues = kMaxLanes;
    if (const char* env = getenv("UGDS_NVMEOF_DIRECT_QUEUES")) t.nr_io_queues = std::max(1, atoi(env));
    ugds_capsule_ctrl* c = nullptr;
    if (int r = ugds_capsule_connect(t, &c)) return r;
    if (ugds_capsule_capacity(c) != capacity || ugds_capsule_block_size(c) != ep->block_size) {
        ugds_capsule_disconnect(c);
        return -EPROTO;
    }
    ep->direct = c;
    return 0;
#else
    (void)ep;
    (void)capacity;
    return -ENOTSUP;
#endif
}

// on_lane == false: uGDSRead/Write. Copy and block I/O on this thread.
// on_lane == true:  uGDSReadAsync/WriteAsync. The CUDA host callback must
// not call cudaMemcpy, so the lane thread does both and we wait here.
ssize_t ugds_nvmeof_io(ugds_nvmeof_ep* ep, void* buf, size_t size, off_t file_offset,
                       off_t buf_offset, bool is_write, bool on_lane) {
    if (!ep->aligned(file_offset, size, buf_offset)) return -EINVAL;
    void* gpu = (char*)buf + buf_offset;
#ifdef UGDS_HAVE_CAPSULE
    // Direct and not inside a CUDA host callback: the initiator only calls
    // CUDA to look up or register the buffer, which is allowed here. If the
    // direct controller has stopped, use the kernel path below; the kernel
    // initiator reconnects on its own.
    if (ep->direct && !on_lane && ugds_capsule_alive(ep->direct)) {
        ssize_t r = ugds_capsule_io(ep->direct, gpu, size, file_offset, is_write);
        if (r >= 0 || ugds_capsule_alive(ep->direct)) return r;
    }
#endif
    // Bounce path on the calling thread: copy GPU->host (write), one
    // pread/pwrite per 16 MiB piece, copy host->GPU (read).
    if (!on_lane) {
        thread_local Bounce tls(false);
        for (size_t done = 0; done < size;) {
            size_t n = std::min(size - done, kSyncPiece);
            void* host = tls.get(n);
            if (!host) return -ENOMEM;
            char* g = (char*)gpu + done;
            if (is_write && copy(host, g, n, false, nullptr)) return -EIO;
            ssize_t r = block_io(ep->fd, host, n, file_offset + (off_t)done, is_write);
            if (r < 0) return r;
            if (!is_write && copy(g, host, n, true, nullptr)) return -EIO;
            done += n;
        }
        return (ssize_t)size;
    }
    // Lane path (Async): w and op live on this stack frame, so return only
    // after the last chunk has called chunk_done().
    if (!ep->start()) return -ENOMEM;
    Waiter w;
    Op op;
    op.w = &w;
    ep->dispatch(&op, gpu, size, file_offset, is_write);
    std::unique_lock<std::mutex> g(w.m);
    w.cv.wait(g, [&] { return op.finished; });
    int e = op.err.load();
    return e ? e : (ssize_t)size;
}

// ---- BatchIO --------------------------------------------------------------

// A remote batch. Lives from uGDSBatchIOSetUp until uGDSBatchIODestroy.
//   tag   first field, 'RBN1' (see ugds_nvmeof_is_batch)
//   keep  shared_ptr that keeps the HandleState alive for the batch's
//         lifetime, same as BatchState::hs_sp on the local path
//   ops   one slot per possible entry (capacity from SetUp); `used` is how
//         many the current wave occupies. Guarded by w.m, except the atomic
//         fields lanes update through chunk_done().
namespace {
struct RemoteBatch {
    uint32_t tag = kBatchTag;
    ugds_nvmeof_ep* ep = nullptr;
    HandleState* hs = nullptr;
    std::shared_ptr<HandleState> keep;
    unsigned used = 0;
    Waiter w;
    std::vector<Op> ops;
    explicit RemoteBatch(unsigned n) : ops(n) {}
};
}  // namespace

bool ugds_nvmeof_is_batch(const void* batch) {
    return *static_cast<const uint32_t*>(batch) == kBatchTag;
}

// Same contract as the local SetUp: one batch per handle at a time
// (batch_active), the handle reference taken by the caller is released by
// Destroy, not here.
uGDSError_t ugds_nvmeof_batch_setup(void** out, HandleState* hs,
                                    std::shared_ptr<HandleState> keep, unsigned nr) {
    if (!hs->remote_ep->start()) return make_error(UGDS_INTERNAL_ERROR);
    bool idle = false;
    if (!hs->batch_active.compare_exchange_strong(idle, true))
        return make_error(UGDS_INVALID_VALUE);
    auto* b = new (std::nothrow) RemoteBatch(nr);
    if (!b) {
        hs->batch_active.store(false);
        return make_error(UGDS_OUT_OF_MEMORY);
    }
    b->ep = hs->remote_ep;
    b->hs = hs;
    b->keep = std::move(keep);
    *out = b;
    return UGDS_OK;
}

// All entries are validated before any state changes, so a rejected Submit
// leaves the batch as it was. Accepted entries are dispatched outside w.m:
// lanes take w.m in chunk_done(), and a fast lane must not wait on us.
uGDSError_t ugds_nvmeof_batch_submit(void* batch, unsigned nr, uGDSIOParams_t* iocb) {
    auto* b = static_cast<RemoteBatch*>(batch);
    if (b->hs->closing.load(std::memory_order_acquire)) return make_error(UGDS_BUSY);
    for (unsigned i = 0; i < nr; ++i) {
        const uGDSIOParams_t& p = iocb[i];
        if (!p.devPtr_base || !b->ep->aligned(p.file_offset, p.size, p.devPtr_offset) ||
            (p.opcode != UGDS_READ && p.opcode != UGDS_WRITE))
            return make_error(UGDS_INVALID_VALUE);
    }
    unsigned base;
    {
        std::lock_guard<std::mutex> g(b->w.m);
        // The previous wave has been fully collected. Reuse the Op slots
        // instead of requiring the caller to destroy and set up again.
        if (b->used > 0 && b->w.done == b->used &&
            std::all_of(b->ops.begin(), b->ops.begin() + b->used,
                        [](const Op& o) { return o.reported; }))
            b->used = b->w.done = 0;
        if (b->used + nr > b->ops.size()) return make_error(UGDS_BATCH_CAPACITY_EXCEEDED);
        base = b->used;
        b->used += nr;
        for (unsigned i = 0; i < nr; ++i) {
            Op& op = b->ops[base + i];
            op.w = &b->w;
            op.err.store(0);
            op.cookie = iocb[i].cookie;
            op.size = iocb[i].size;
            op.finished = op.reported = false;
        }
    }
    for (unsigned i = 0; i < nr; ++i) {
        const uGDSIOParams_t& p = iocb[i];
        b->ep->dispatch(&b->ops[base + i], (char*)p.devPtr_base + p.devPtr_offset,
                        p.size, p.file_offset, p.opcode == UGDS_WRITE);
    }
    return UGDS_OK;
}

uGDSError_t ugds_nvmeof_batch_status(void* batch, unsigned min_nr, unsigned* nr,
                                     uGDSIOEvents_t* events, struct timespec* timeout) {
    auto* b = static_cast<RemoteBatch*>(batch);
    unsigned max_events = *nr > 0 ? *nr : (unsigned)b->ops.size();
    std::unique_lock<std::mutex> g(b->w.m);
    // Wait until min_nr completions exist that this call has not already
    // returned. A short wait still returns whatever is finished, including
    // zero; the caller polls again. timeout == NULL waits indefinitely.
    auto ready = [&] {
        unsigned reported = 0;
        for (unsigned i = 0; i < b->used; ++i) reported += b->ops[i].reported;
        return b->w.done - reported >= std::min(min_nr, b->used - reported);
    };
    if (timeout) {
        auto d = std::chrono::seconds(timeout->tv_sec) + std::chrono::nanoseconds(timeout->tv_nsec);
        b->w.cv.wait_for(g, d, ready);
    } else {
        b->w.cv.wait(g, ready);
    }
    unsigned n = 0;
    for (unsigned i = 0; i < b->used && n < max_events; ++i) {
        Op& op = b->ops[i];
        if (!op.finished || op.reported) continue;
        int e = op.err.load();
        events[n].cookie = op.cookie;
        events[n].status = e ? UGDS_BATCH_FAILED : UGDS_BATCH_COMPLETE;
        events[n].ret = e ? e : (ssize_t)op.size;
        op.reported = true;
        n++;
    }
    *nr = n;
    return UGDS_OK;
}

// Waits for every submitted entry (lanes hold pointers into b->ops), then
// releases the handle like the local Destroy does.
void ugds_nvmeof_batch_destroy(void* batch) {
    auto* b = static_cast<RemoteBatch*>(batch);
    {
        std::unique_lock<std::mutex> g(b->w.m);
        b->w.cv.wait(g, [&] { return b->w.done == b->used; });
    }
    b->hs->batch_active.store(false);
    handle_release(b->hs);
    delete b;
}
