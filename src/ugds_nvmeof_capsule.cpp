// User-space NVMe-oF/RDMA initiator with GPU-resident data buffers.
//
// Wire format follows the NVMe over Fabrics spec and the Linux host
// (drivers/nvme/host/rdma.c, fabrics.c):
//  - One RC QP per NVMe queue, set up through RDMA CM. The CM request
//    carries {recfmt, qid, hrqsize, hsqsize}.
//  - A command capsule is one RDMA SEND of the 64-byte SQE. Every command
//    uses a keyed SGL data block (PSDT = 01b), even when it moves no data.
//  - The target answers with one RDMA SEND of the 16-byte CQE into a
//    receive we posted earlier. CQEs are matched to commands by CID.
//  - Data never passes through our CPU: for a read the target RDMA-WRITEs
//    into the SGL address, for a write it RDMA-READs from it. The SGL holds
//    a GPU virtual address and the rkey of a dma-buf memory region.
//
// Controller lifecycle (ugds_capsule_connect / ugds_capsule_disconnect):
//
//   1. Admin queue: RDMA CM connect, then Fabrics Connect (qid 0,
//      cntlid 0xffff). The CQE returns the target's cntlid.
//   2. Property Get CAP (MQES caps the I/O queue depth, TO bounds the
//      ready wait), Property Set CC.EN, poll CSTS.RDY.
//   3. Identify Namespace: capacity and LBA size.
//   4. Each I/O queue: RDMA CM connect, then Fabrics Connect with the
//      cntlid from step 1. A queue that fails is rebuilt (see
//      kConnectAttempts); the others are kept.
//   5. Keep-alive thread: Keep Alive on the admin queue every KATO / 2.
//   ...I/O...
//   6. Disconnect: stop keep-alive, Property Set CC.SHN, tear down QPs and
//      memory regions, then the PD, then the CM ids.
//
// Threads and locks:
//   - Queue::m is held for a whole submission (post + wait). A queue never
//     has commands from two submissions in flight.
//   - The admin queue is shared by setup, keep-alive and disconnect; its
//     lock serializes them.
//   - mr_m guards the GPU registration cache.
//   - `dead` is the only cross-queue state on the I/O path.
//
// Failure model: any transport error or command timeout stops the
// controller for good (mark_dead): every QP is moved to the error state at
// once so the target cannot write into GPU memory any more, and later I/O
// returns -EIO immediately. There is no reconnect here; the uGDS layer
// (ugds_nvmeof.cpp) switches the handle back to the kernel path, which
// reconnects on its own.
#include "ugds_nvmeof_capsule.h"

#include <cuda.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <arpa/inet.h>
#include <poll.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <dirent.h>

namespace {

// NVM command set and admin opcodes (NVMe base spec), fabrics opcode and
// fctype values (NVMe-oF spec).
constexpr uint8_t kOpWrite = 0x01;
constexpr uint8_t kOpRead = 0x02;
constexpr uint8_t kOpIdentify = 0x06;
constexpr uint8_t kOpKeepAlive = 0x18;
constexpr uint8_t kOpFabrics = 0x7f;
constexpr uint8_t kFctPropSet = 0x00;
constexpr uint8_t kFctConnect = 0x01;
constexpr uint8_t kFctPropGet = 0x04;

// SQE byte 1, PSDT = 01b: the data pointer is an SGL. Fabrics requires it.
constexpr uint8_t kFlagSgl = 0x40;
// SGL descriptor type 4 (keyed data block), subtype 0 (address).
constexpr uint8_t kSglKeyedData = 0x40;

constexpr uint32_t kRegCap = 0x00;
constexpr uint32_t kRegCc = 0x14;
constexpr uint32_t kRegCsts = 0x1c;
constexpr uint64_t kCcEnable = 1u << 0;
constexpr uint64_t kCcShutdownNormal = 1u << 14;
constexpr uint64_t kCcIoSqes = 6u << 16;  // 2^6 = 64-byte SQE
constexpr uint64_t kCcIoCqes = 4u << 20;  // 2^4 = 16-byte CQE

constexpr int kAdminDepth = 32;  // NVME_AQ_DEPTH in the Linux host
constexpr int kMaxDepth = 64;    // CID low 6 bits index the slot
// Keep-alive timeout sent in the admin Connect. The target drops the
// controller if it hears nothing for this long; we send Keep Alive every
// half of it.
constexpr uint32_t kKatoMs = 10000;
// A submission is cut into pieces (a power of two within these bounds) so
// that its commands run in parallel. On the GP target one large command is
// far slower than the same bytes as parallel pieces: QD1 128 KiB takes
// 551 us as one command and 181 us as 8 x 16 KiB; 1 MiB takes 1889 us as one
// and 497 us as 16 x 64 KiB. When many submissions run at once the queues
// are already full, and splitting further only adds commands, so the
// controller aims for about kInflightBudget commands in total.
constexpr size_t kMinPiece = 16u << 10;
constexpr size_t kMaxPiece = 1u << 20;  // nvme-rdma max_sectors_kb default
constexpr int kInflightBudget = 64;
constexpr int kCmTimeoutMs = 5000;
constexpr int kAdminTimeoutMs = 10000;
// The GP5000 target occasionally never answers one queue's Fabrics Connect
// (the kernel host hit the same thing on a reconnect: "Connect command
// failed" on one QID). A normal answer takes ~100 ms, so give up on that
// queue after a few seconds and build it again on a new connection.
constexpr int kConnectTimeoutMs = 3000;
constexpr int kConnectAttempts = 4;
constexpr int kIoTimeoutMs = 30000;  // same as nvme_core io_timeout

// 64-byte submission queue entry as raw bytes. Fields are written with
// put16/32/64 at their spec offsets (little-endian host assumed, as on the
// rest of uGDS):
//    0  opcode            1  flags (PSDT)      2  CID
//    4  NSID, or fctype for fabrics commands
//   24  SGL descriptor (16 bytes, see set_sgl)
//   40  CDW10 ..  63  CDW15  (command specific)
struct Sqe {
    uint8_t b[64];
};

// 16-byte completion queue entry. dw0/dw1 carry the command's result (a
// Property Get returns up to 8 bytes there).
struct Cqe {
    uint32_t dw0;
    uint32_t dw1;
    uint16_t sq_head;
    uint16_t sq_id;
    uint16_t cid;
    uint16_t status;  // bit 0 is the phase tag
};
static_assert(sizeof(Cqe) == 16, "NVMe CQE is 16 bytes");

void put16(Sqe& s, int off, uint16_t v) { memcpy(s.b + off, &v, 2); }
void put32(Sqe& s, int off, uint32_t v) { memcpy(s.b + off, &v, 4); }
void put64(Sqe& s, int off, uint64_t v) { memcpy(s.b + off, &v, 8); }

// Zeroed SQE with opcode and PSDT set; CID is filled in by submit().
Sqe make_cmd(uint8_t opcode) {
    Sqe s{};
    s.b[0] = opcode;
    s.b[1] = kFlagSgl;
    return s;
}

// Keyed SGL data block in bytes 24..39: address (8), length (3), key (4),
// type (1). addr is in the memory region's IOVA space: a host pointer for
// admin_buf, a GPU virtual address for dma-buf regions. Commands without
// data use addr = len = key = 0, as the Linux host does.
void set_sgl(Sqe& s, uint64_t addr, uint32_t len, uint32_t rkey) {
    put64(s, 24, addr);
    s.b[32] = len & 0xff;
    s.b[33] = (len >> 8) & 0xff;
    s.b[34] = (len >> 16) & 0xff;
    put32(s, 35, rkey);
    s.b[39] = kSglKeyedData;
}

// Status field without the phase bit; 0 means success.
uint16_t cqe_status(const Cqe& c) { return c.status >> 1; }

// One NVMe queue = one RC QP with its own CM id, event channel and CQ (sends
// and receives share the CQ). `depth` receives are posted at all times, one
// per rsp slot; a receive is re-posted as soon as its CQE has been copied.
// The SQE slots are reused by every submission, which is safe because a
// submission waits for all of its send completions before releasing m.
struct Queue {
    int qid = 0;
    int depth = 0;
    rdma_event_channel* ch = nullptr;
    rdma_cm_id* id = nullptr;
    ibv_cq* cq = nullptr;
    uint8_t* sqe = nullptr;  // depth * 64 bytes, one slot per command
    uint8_t* rsp = nullptr;  // depth * 16 bytes, one slot per receive
    ibv_mr* sqe_mr = nullptr;
    ibv_mr* rsp_mr = nullptr;
    uint16_t epoch = 0;  // CID high bits; changes every submit
    std::mutex m;
};

// One NIC registration covering a whole CUDA allocation [base, base + len).
struct GpuMr {
    uintptr_t base;
    size_t len;
    unsigned long long buffer_id;  // CU_POINTER_ATTRIBUTE_BUFFER_ID; never reused
    ibv_mr* mr;
};

// Wait up to timeout_ms for the next CM event on ch and require it to be
// `want` with status 0. rdma_get_cm_event() itself has no timeout, hence
// the poll() first. A reject from the target is logged with its NVMe/RDMA
// status so a misconfigured host NQN or queue size is visible.
int wait_cm(rdma_event_channel* ch, rdma_cm_event_type want, int timeout_ms) {
    pollfd p{ch->fd, POLLIN, 0};
    if (poll(&p, 1, timeout_ms) <= 0) {
        fprintf(stderr, "uGDS capsule: no CM event (wanted %s)\n", rdma_event_str(want));
        return -ETIMEDOUT;
    }
    rdma_cm_event* ev = nullptr;
    if (rdma_get_cm_event(ch, &ev)) return -errno;
    bool ok = ev->event == want && ev->status == 0;
    if (!ok) {
        fprintf(stderr, "uGDS capsule: CM %s status %d (wanted %s)", rdma_event_str(ev->event),
                ev->status, rdma_event_str(want));
        // NVMe/RDMA reject private data: {recfmt, sts}.
        if (ev->event == RDMA_CM_EVENT_REJECTED && ev->param.conn.private_data_len >= 4) {
            uint16_t sts;
            memcpy(&sts, (const uint8_t*)ev->param.conn.private_data + 2, 2);
            fprintf(stderr, " nvme reject sts=%u", sts);
        }
        fprintf(stderr, "\n");
    }
    rdma_ack_cm_event(ev);
    return ok ? 0 : -ECONNREFUSED;
}

// Contents of a small sysfs file without trailing newline/space; "" if
// unreadable.
std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[512] = {};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    std::string s(buf, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// Value of key in "a=1,b=2" (the format of /sys/class/nvme/*/address).
std::string kv(const std::string& list, const std::string& key) {
    size_t p = 0;
    while (p < list.size()) {
        size_t e = list.find(',', p);
        if (e == std::string::npos) e = list.size();
        std::string item = list.substr(p, e - p);
        if (item.compare(0, key.size() + 1, key + "=") == 0) return item.substr(key.size() + 1);
        p = e + 1;
    }
    return "";
}

// Canonical UUID text (sysfs hostid) to its 16 bytes in textual order,
// which is the byte order of the Connect data's HOSTID field.
int parse_uuid(const std::string& s, uint8_t out[16]) {
    int n = 0;
    for (size_t i = 0; i + 1 < s.size() && n < 16;) {
        if (s[i] == '-') { ++i; continue; }
        unsigned v;
        if (sscanf(s.c_str() + i, "%2x", &v) != 1) return -EINVAL;
        out[n++] = (uint8_t)v;
        i += 2;
    }
    return n == 16 ? 0 : -EINVAL;
}

}  // namespace

// Everything below except `dead`, `rr`, `active`, the MR cache and the
// keep-alive fields is written only during connect, before the controller
// is returned, and read-only afterwards.
//   ctx, pd     taken from the admin queue's CM id; every queue and MR uses
//               this one PD so a GPU registration works on any queue
//   rd_atom     max_qp_rd_atom, offered as responder_resources: the target
//               RDMA-reads from us for writes
//   admin_buf   host buffer for Connect data and Identify; also referenced
//               by I/O-queue Connects during setup
struct ugds_capsule_ctrl {
    ugds_capsule_target t;
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    int rd_atom = 1;
    std::vector<std::unique_ptr<Queue>> qs;  // qs[0] is the admin queue
    uint8_t* admin_buf = nullptr;            // 4 KiB host buffer for admin data
    ibv_mr* admin_mr = nullptr;
    uint16_t cntlid = 0;
    int io_depth = 0;
    uint64_t capacity = 0;
    uint32_t lba_shift = 0;
    std::atomic<bool> dead{false};
    std::atomic<uint32_t> rr{0};
    std::atomic<int> active{0};  // submissions currently holding a queue

    std::mutex mr_m;
    std::vector<GpuMr> mrs;

    std::mutex ka_m;
    std::condition_variable ka_cv;
    bool ka_stop = false;
    std::thread ka;
};

namespace {

// Stop the controller for good. Every QP goes to the error state at once, so
// the target can no longer RDMA into GPU buffers whose I/O already returned
// an error to the caller.
void mark_dead(ugds_capsule_ctrl* c, const char* why) {
    if (c->dead.exchange(true)) return;
    fprintf(stderr, "uGDS capsule: controller stopped (%s)\n", why);
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_ERR;
    for (auto& q : c->qs)
        if (q->id && q->id->qp) ibv_modify_qp(q->id->qp, &a, IBV_QP_STATE);
}

// Post n (<= q->depth) commands on q and wait for all n CQEs; out[i] gets
// the CQE of cmds[i]. Caller holds q->m. A failure marks the whole
// controller dead unless fatal is false (queue bring-up, where only this
// queue is retried).
//
// CID = epoch << 6 | slot. The epoch advances on every submission, so a CQE
// that arrives late for an earlier, abandoned submission does not match and
// is dropped instead of being taken for a current command.
//
// Every send is signaled and counted, so when this returns the target has
// acknowledged every SQE and the SQE slots can be reused.
int submit(ugds_capsule_ctrl* c, Queue* q, Sqe* cmds, int n, Cqe* out, int timeout_ms,
           bool fatal = true) {
    if (c->dead.load(std::memory_order_acquire)) return -EIO;
    uint16_t epoch = q->epoch++ & 0x3ff;
    ibv_sge sge[kMaxDepth];
    ibv_send_wr wr[kMaxDepth];
    for (int i = 0; i < n; ++i) {
        put16(cmds[i], 2, (uint16_t)(epoch << 6 | i));
        memcpy(q->sqe + i * 64, cmds[i].b, 64);
        sge[i] = {(uint64_t)(q->sqe + i * 64), 64, q->sqe_mr->lkey};
        wr[i] = {};
        wr[i].wr_id = i;
        wr[i].opcode = IBV_WR_SEND;
        wr[i].send_flags = IBV_SEND_SIGNALED;
        wr[i].sg_list = &sge[i];
        wr[i].num_sge = 1;
        wr[i].next = i + 1 < n ? &wr[i + 1] : nullptr;
    }
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(q->id->qp, wr, &bad)) {
        if (fatal) mark_dead(c, "post_send failed");
        return -EIO;
    }

    bool got[kMaxDepth] = {};
    int sends = 0, rsps = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    unsigned idle = 0;
    while (sends < n || rsps < n) {
        ibv_wc wc[16];
        int k = ibv_poll_cq(q->cq, 16, wc);
        if (k < 0) {
            if (fatal) mark_dead(c, "poll_cq failed");
            return -EIO;
        }
        if (k == 0) {
            if ((++idle & 0xfff) == 0 && std::chrono::steady_clock::now() > deadline) {
                fprintf(stderr, "uGDS capsule: qid %d timed out (%d/%d CQEs)\n", q->qid, rsps, n);
                if (fatal) mark_dead(c, "command timeout");
                return -ETIMEDOUT;
            }
            // Spin briefly, then give the CPU away between polls. When there
            // are more submitting threads than cores, pure spinning halves
            // throughput; with a core to spare, sched_yield returns at once.
            if (idle > 64) sched_yield();
            continue;
        }
        for (int j = 0; j < k; ++j) {
            if (wc[j].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "uGDS capsule: qid %d wc %s\n", q->qid,
                        ibv_wc_status_str(wc[j].status));
                if (fatal) mark_dead(c, "work completion error");
                return -EIO;
            }
            if (wc[j].opcode == IBV_WC_SEND) {
                sends++;
                continue;
            }
            int slot_r = (int)wc[j].wr_id;
            Cqe e;
            memcpy(&e, q->rsp + slot_r * 16, 16);
            ibv_sge rs{(uint64_t)(q->rsp + slot_r * 16), 16, q->rsp_mr->lkey};
            ibv_recv_wr rw{};
            rw.wr_id = slot_r;
            rw.sg_list = &rs;
            rw.num_sge = 1;
            ibv_recv_wr* rbad = nullptr;
            if (ibv_post_recv(q->id->qp, &rw, &rbad)) {
                if (fatal) mark_dead(c, "post_recv failed");
                return -EIO;
            }
            int slot = e.cid & 63;
            if ((e.cid >> 6) != epoch || slot >= n || got[slot]) {
                fprintf(stderr, "uGDS capsule: qid %d stray CQE cid 0x%x\n", q->qid, e.cid);
                continue;
            }
            got[slot] = true;
            out[slot] = e;
            rsps++;
        }
    }
    return 0;
}

// One admin command, waiting for its CQE. A non-zero NVMe status is -EIO
// but does not stop the controller; a transport failure does.
int admin_cmd(ugds_capsule_ctrl* c, Sqe s, Cqe* out) {
    Queue* q = c->qs[0].get();
    std::lock_guard<std::mutex> g(q->m);
    int r = submit(c, q, &s, 1, out, kAdminTimeoutMs);
    if (r) return r;
    if (cqe_status(*out)) {
        fprintf(stderr, "uGDS capsule: admin opcode 0x%02x fctype 0x%02x status 0x%x\n", s.b[0],
                s.b[4], cqe_status(*out));
        return -EIO;
    }
    return 0;
}

// Fabrics Property Get/Set: the controller registers (CAP, CC, CSTS) that a
// PCIe host would read from BAR0. Byte 40 is ATTRIB (0 = 4 bytes,
// 1 = 8 bytes), 44 is the register offset, 48 the value to set. A Get
// returns the value in CQE dw0 (and dw1 for 8 bytes).
int prop_get(ugds_capsule_ctrl* c, uint32_t reg, bool eight, uint64_t* v) {
    Sqe s = make_cmd(kOpFabrics);
    s.b[4] = kFctPropGet;
    set_sgl(s, 0, 0, 0);
    s.b[40] = eight ? 1 : 0;
    put32(s, 44, reg);
    Cqe e;
    int r = admin_cmd(c, s, &e);
    if (!r) *v = (uint64_t)e.dw0 | ((uint64_t)e.dw1 << 32);
    return r;
}

int prop_set(ugds_capsule_ctrl* c, uint32_t reg, uint32_t v) {
    Sqe s = make_cmd(kOpFabrics);
    s.b[4] = kFctPropSet;
    set_sgl(s, 0, 0, 0);
    s.b[40] = 0;
    put32(s, 44, reg);
    put64(s, 48, v);
    Cqe e;
    return admin_cmd(c, s, &e);
}

// Fabrics Connect for queue q, sent on q itself (the spec requires each
// queue to be bound by its own Connect).
//
// Connect data, 1 KiB, in admin_buf (the target RDMA-reads it):
//     0  HOSTID (16)     16  CNTLID (2): 0xffff = "give me a new
//                                        controller" on qid 0, the
//                                        returned cntlid on I/O queues
//   256  SUBNQN (256)   512  HOSTNQN (256)   768  reserved (256)
// SQE: CDW10 = RECFMT | QID << 16, CDW11 = SQSIZE (0-based), CDW12 = KATO
// in ms (admin queue only).
//
// The submit is non-fatal and uses kConnectTimeoutMs: a queue whose Connect
// is never answered is retried by the caller rather than killing the
// controller.
int fabrics_connect(ugds_capsule_ctrl* c, Queue* q) {
    uint8_t* d = c->admin_buf;
    memset(d, 0, 1024);
    memcpy(d, c->t.hostid, 16);
    uint16_t cntlid = q->qid == 0 ? 0xffff : c->cntlid;
    memcpy(d + 16, &cntlid, 2);
    snprintf((char*)d + 256, 256, "%s", c->t.subnqn.c_str());
    snprintf((char*)d + 512, 256, "%s", c->t.hostnqn.c_str());

    Sqe s = make_cmd(kOpFabrics);
    s.b[4] = kFctConnect;
    set_sgl(s, (uint64_t)d, 1024, c->admin_mr->rkey);
    put16(s, 40, 0);  // recfmt
    put16(s, 42, (uint16_t)q->qid);
    put16(s, 44, (uint16_t)(q->depth - 1));
    put32(s, 48, q->qid == 0 ? kKatoMs : 0);
    Cqe e;
    std::lock_guard<std::mutex> g(q->m);
    int r = submit(c, q, &s, 1, &e, kConnectTimeoutMs, /*fatal=*/false);
    if (r) return r;
    if (cqe_status(e)) {
        fprintf(stderr, "uGDS capsule: Connect qid %d status 0x%x\n", q->qid, cqe_status(e));
        return -ECONNREFUSED;
    }
    if (q->qid == 0) c->cntlid = e.dw0 & 0xffff;
    return 0;
}

// Transport setup for one queue, up to RDMA_CM_EVENT_ESTABLISHED:
// resolve address and route, create (on the first queue) the shared PD and
// admin buffer, create CQ and QP, register the SQE/CQE slots, pre-post
// `depth` receives, then connect with the NVMe/RDMA private data. On
// failure the caller cleans up with queue_close() and rdma_destroy_id().
int queue_open(ugds_capsule_ctrl* c, Queue* q, int qid, int depth) {
    q->qid = qid;
    q->depth = depth;
    q->ch = rdma_create_event_channel();
    if (!q->ch) return -errno;
    if (rdma_create_id(q->ch, &q->id, nullptr, RDMA_PS_TCP)) return -errno;

    sockaddr_in dst{}, src{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons((uint16_t)atoi(c->t.trsvcid.c_str()));
    if (inet_pton(AF_INET, c->t.traddr.c_str(), &dst.sin_addr) != 1) return -EINVAL;
    sockaddr* srcp = nullptr;
    if (!c->t.host_traddr.empty()) {
        src.sin_family = AF_INET;
        if (inet_pton(AF_INET, c->t.host_traddr.c_str(), &src.sin_addr) != 1) return -EINVAL;
        srcp = (sockaddr*)&src;
    }
    if (rdma_resolve_addr(q->id, srcp, (sockaddr*)&dst, kCmTimeoutMs)) return -errno;
    if (int r = wait_cm(q->ch, RDMA_CM_EVENT_ADDR_RESOLVED, kCmTimeoutMs + 1000)) return r;
    if (rdma_resolve_route(q->id, kCmTimeoutMs)) return -errno;
    if (int r = wait_cm(q->ch, RDMA_CM_EVENT_ROUTE_RESOLVED, kCmTimeoutMs + 1000)) return r;

    // All queues share one PD so a GPU memory region works on every QP.
    if (!c->ctx) {
        c->ctx = q->id->verbs;
        c->pd = ibv_alloc_pd(c->ctx);
        if (!c->pd) return -ENOMEM;
        ibv_device_attr attr{};
        if (ibv_query_device(c->ctx, &attr) == 0) c->rd_atom = attr.max_qp_rd_atom;
        c->admin_buf = (uint8_t*)aligned_alloc(4096, 4096);
        if (!c->admin_buf) return -ENOMEM;
        c->admin_mr = ibv_reg_mr(c->pd, c->admin_buf, 4096,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                                     IBV_ACCESS_REMOTE_WRITE);
        if (!c->admin_mr) return -errno;
    } else if (q->id->verbs != c->ctx) {
        return -EXDEV;
    }

    // A submit waits for all of its sends and CQEs before returning, so the
    // CQ only has to hold two completions per command in flight. Admin
    // commands are issued one at a time under the queue lock.
    int inflight = qid == 0 ? 1 : depth;
    q->cq = ibv_create_cq(c->ctx, 2 * inflight + 2, nullptr, nullptr, 0);
    if (!q->cq) return -errno;
    ibv_qp_init_attr qa{};
    qa.qp_type = IBV_QPT_RC;
    qa.send_cq = q->cq;
    qa.recv_cq = q->cq;
    qa.cap.max_send_wr = depth;
    qa.cap.max_recv_wr = depth;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    if (rdma_create_qp(q->id, c->pd, &qa)) return -errno;

    q->sqe = (uint8_t*)aligned_alloc(64, (size_t)depth * 64);
    q->rsp = (uint8_t*)aligned_alloc(64, (size_t)depth * 16);
    if (!q->sqe || !q->rsp) return -ENOMEM;
    q->sqe_mr = ibv_reg_mr(c->pd, q->sqe, (size_t)depth * 64, IBV_ACCESS_LOCAL_WRITE);
    q->rsp_mr = ibv_reg_mr(c->pd, q->rsp, (size_t)depth * 16, IBV_ACCESS_LOCAL_WRITE);
    if (!q->sqe_mr || !q->rsp_mr) return -errno;
    for (int i = 0; i < depth; ++i) {
        ibv_sge rs{(uint64_t)(q->rsp + i * 16), 16, q->rsp_mr->lkey};
        ibv_recv_wr rw{};
        rw.wr_id = i;
        rw.sg_list = &rs;
        rw.num_sge = 1;
        ibv_recv_wr* rbad = nullptr;
        if (ibv_post_recv(q->id->qp, &rw, &rbad)) return -EIO;
    }

    // NVMe/RDMA CM request private data (32 bytes): record format 0, queue
    // id, host receive queue size (receives we posted), host send queue
    // size (0-based SQ depth). Same values the Linux host sends.
    struct {
        uint16_t recfmt, qid, hrqsize, hsqsize;
        uint8_t rsvd[24];
    } priv{};
    priv.qid = (uint16_t)qid;
    priv.hrqsize = (uint16_t)depth;
    priv.hsqsize = (uint16_t)(depth - 1);
    rdma_conn_param param{};
    param.private_data = &priv;
    param.private_data_len = sizeof(priv);
    param.flow_control = 1;
    param.responder_resources = (uint8_t)std::min(c->rd_atom, 255);
    param.retry_count = 7;
    param.rnr_retry_count = 7;
    if (rdma_connect(q->id, &param)) return -errno;
    return wait_cm(q->ch, RDMA_CM_EVENT_ESTABLISHED, kCmTimeoutMs);
}

// Disconnect and free a queue's QP, CQ and slot buffers. Keeps the CM id
// and event channel: the last CM id must outlive the PD (see disconnect).
void queue_close(Queue* q) {
    if (q->id && q->id->qp) {
        rdma_disconnect(q->id);
        rdma_destroy_qp(q->id);
    }
    if (q->sqe_mr) ibv_dereg_mr(q->sqe_mr);
    if (q->rsp_mr) ibv_dereg_mr(q->rsp_mr);
    if (q->cq) ibv_destroy_cq(q->cq);
    free(q->sqe);
    free(q->rsp);
    q->sqe_mr = q->rsp_mr = nullptr;
    q->cq = nullptr;
    q->sqe = q->rsp = nullptr;
}

// Identify Namespace (CNS 0) into admin_buf. Uses NSZE (bytes 0..7) times
// the data size of the active LBA format (FLBAS bits 3:0 at byte 26 index
// the LBA format table at byte 128, 4 bytes per entry, LBADS in byte 2),
// which is what the kernel reports as BLKGETSIZE64.
int identify_ns(ugds_capsule_ctrl* c) {
    Sqe s = make_cmd(kOpIdentify);
    put32(s, 4, c->t.nsid);
    set_sgl(s, (uint64_t)c->admin_buf, 4096, c->admin_mr->rkey);
    put32(s, 40, 0);  // CNS 0: namespace data structure
    memset(c->admin_buf, 0, 4096);
    Cqe e;
    if (int r = admin_cmd(c, s, &e)) return r;
    uint64_t nsze;
    memcpy(&nsze, c->admin_buf, 8);
    uint8_t flbas = c->admin_buf[26] & 0x0f;
    uint8_t lbads = c->admin_buf[128 + flbas * 4 + 2];
    if (nsze == 0 || lbads < 9 || lbads > 16) return -EPROTO;
    c->lba_shift = lbads;
    c->capacity = nsze << lbads;
    return 0;
}

// Keep-alive thread body. Sleeps KATO / 2 on ka_cv so disconnect can wake
// it at once. A Keep Alive that times out stops the controller through
// admin_cmd's fatal submit; one that returns an NVMe error is only logged.
void keep_alive(ugds_capsule_ctrl* c) {
    std::unique_lock<std::mutex> lk(c->ka_m);
    while (!c->ka_cv.wait_for(lk, std::chrono::milliseconds(kKatoMs / 2), [&] { return c->ka_stop; })) {
        lk.unlock();
        Sqe s = make_cmd(kOpKeepAlive);
        set_sgl(s, 0, 0, 0);
        Cqe e;
        if (admin_cmd(c, s, &e)) fprintf(stderr, "uGDS capsule: keep-alive failed\n");
        lk.lock();
    }
}

// The registration covering [p, p + n), if it belongs to the allocation that
// lives at p now. A registration for a freed allocation is dropped.
ibv_mr* find_mr(ugds_capsule_ctrl* c, uintptr_t p, size_t n, unsigned long long buffer_id) {
    std::lock_guard<std::mutex> g(c->mr_m);
    for (auto it = c->mrs.begin(); it != c->mrs.end(); ++it) {
        if (p < it->base || p >= it->base + it->len) continue;
        if (it->buffer_id != buffer_id) {
            ibv_dereg_mr(it->mr);
            c->mrs.erase(it);
            return nullptr;
        }
        return p + n <= it->base + it->len ? it->mr : nullptr;
    }
    return nullptr;
}

// CUDA's per-allocation ID for p, or 0 if p is not device memory. Works
// without a current context (unified addressing), so lane threads and
// caller threads that never touched CUDA can call it.
unsigned long long buffer_id_of(const void* p) {
    unsigned long long id = 0;
    if (cuPointerGetAttribute(&id, CU_POINTER_ATTRIBUTE_BUFFER_ID, (CUdeviceptr)p) != CUDA_SUCCESS)
        return 0;
    return id;
}

// MR for [p, p + n), registering the containing allocation on a miss. The
// returned pointer is used after mr_m is dropped; it stays valid because
// only a different buffer ID at the same address (the caller freed the
// buffer, which it must not do with I/O in flight) or disconnect removes it.
int get_mr(ugds_capsule_ctrl* c, void* p, size_t n, ibv_mr** out) {
    unsigned long long id = buffer_id_of(p);
    if (!id) return -EFAULT;  // not CUDA device memory
    if ((*out = find_mr(c, (uintptr_t)p, n, id))) return 0;
    if (int r = ugds_capsule_register(c, p, n)) return r;
    *out = find_mr(c, (uintptr_t)p, n, id);
    return *out ? 0 : -EINVAL;
}

// Lock an I/O queue: the first free one starting from a rotating index, or
// wait on that index if every queue is busy.
Queue* take_queue(ugds_capsule_ctrl* c) {
    size_t nio = c->qs.size() - 1;
    uint32_t start = c->rr.fetch_add(1, std::memory_order_relaxed);
    for (size_t k = 0; k < nio; ++k) {
        Queue* q = c->qs[1 + (start + k) % nio].get();
        if (q->m.try_lock()) return q;
    }
    Queue* q = c->qs[1 + start % nio].get();
    q->m.lock();
    return q;
}

}  // namespace

// fd -> /proc/self/fd link -> /sys/block/<name> -> nsid and controller
// directory -> address ("traddr=...,trsvcid=...[,src_addr=...]"),
// subsysnqn, hostnqn, hostid. Using the kernel's host NQN and host ID means
// the target's host access list already admits us.
int ugds_capsule_target_from_fd(int fd, ugds_capsule_target* t) {
    char link[256] = {};
    std::string self = "/proc/self/fd/" + std::to_string(fd);
    if (readlink(self.c_str(), link, sizeof(link) - 1) <= 0) return -errno;
    const char* name = strrchr(link, '/');
    if (!name) return -EINVAL;
    std::string blk = std::string("/sys/block/") + (name + 1);
    std::string nsid = read_file(blk + "/nsid");
    if (nsid.empty()) return -ENODEV;

    // Multipath head: device -> nvme-subsysN, which lists its controllers.
    // Path device: device -> the controller itself.
    std::string ctrl;
    std::string dev = blk + "/device";
    if (!read_file(dev + "/address").empty()) {
        ctrl = dev;
    } else if (DIR* d = opendir(dev.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string cand = dev + "/" + e->d_name;
            if (strncmp(e->d_name, "nvme", 4) == 0 && read_file(cand + "/transport") == "rdma" &&
                read_file(cand + "/state") == "live") {
                ctrl = cand;
                break;
            }
        }
        closedir(d);
    }
    if (ctrl.empty() || read_file(ctrl + "/transport") != "rdma") return -ENODEV;

    std::string addr = read_file(ctrl + "/address");
    t->traddr = kv(addr, "traddr");
    t->trsvcid = kv(addr, "trsvcid");
    t->host_traddr = kv(addr, "src_addr");
    t->subnqn = read_file(ctrl + "/subsysnqn");
    t->hostnqn = read_file(ctrl + "/hostnqn");
    t->nsid = (uint32_t)strtoul(nsid.c_str(), nullptr, 10);
    if (t->traddr.empty() || t->trsvcid.empty() || t->subnqn.empty() || t->hostnqn.empty())
        return -ENODEV;
    return parse_uuid(read_file(ctrl + "/hostid"), t->hostid);
}

int ugds_capsule_connect(const ugds_capsule_target& t, ugds_capsule_ctrl** out) {
    if (t.nr_io_queues < 1 || t.queue_depth < 2) return -EINVAL;
    auto* c = new (std::nothrow) ugds_capsule_ctrl();
    if (!c) return -ENOMEM;
    c->t = t;
    // Every error path tears down whatever exists so far; disconnect copes
    // with a partly built controller.
    auto fail = [&](int r) {
        ugds_capsule_disconnect(c);
        return r;
    };

    // Steps 1-3 of the lifecycle at the top of this file.
    c->qs.emplace_back(new Queue());
    if (int r = queue_open(c, c->qs[0].get(), 0, kAdminDepth)) return fail(r);
    if (int r = fabrics_connect(c, c->qs[0].get())) return fail(r);

    // CAP.MQES (bits 15:0) is the 0-based maximum queue size; CAP.TO
    // (bits 31:24) is the worst-case ready time in 500 ms units.
    uint64_t cap = 0, csts = 0;
    if (int r = prop_get(c, kRegCap, true, &cap)) return fail(r);
    int mqes = (int)(cap & 0xffff) + 1;
    int ready_ms = (int)((cap >> 24) & 0xff) * 500;
    if (int r = prop_set(c, kRegCc, (uint32_t)(kCcEnable | kCcIoSqes | kCcIoCqes))) return fail(r);
    for (int waited = 0;; waited += 10) {
        if (int r = prop_get(c, kRegCsts, false, &csts)) return fail(r);
        if (csts & 1) break;
        if (waited > std::max(ready_ms, 1000)) return fail(-ETIMEDOUT);
        usleep(10000);
    }
    if (int r = identify_ns(c)) return fail(r);

    // Step 4. Queues are opened one after another; each failed attempt is
    // torn down completely and rebuilt on a fresh CM id.
    c->io_depth = std::min({t.queue_depth, mqes, kMaxDepth});
    for (int i = 1; i <= t.nr_io_queues; ++i) {
        int r = -ETIMEDOUT;
        for (int attempt = 0; attempt < kConnectAttempts && r; ++attempt) {
            auto q = std::make_unique<Queue>();
            r = queue_open(c, q.get(), i, c->io_depth);
            if (!r) r = fabrics_connect(c, q.get());
            if (!r) {
                c->qs.push_back(std::move(q));
                break;
            }
            fprintf(stderr, "uGDS capsule: qid %d setup failed (%d), attempt %d\n", i, r,
                    attempt + 1);
            // The admin queue's cm_id keeps the device context open, so this
            // queue's id can go right away.
            queue_close(q.get());
            if (q->id) rdma_destroy_id(q->id);
            if (q->ch) rdma_destroy_event_channel(q->ch);
        }
        if (r) return fail(r);
    }
    c->ka = std::thread(keep_alive, c);
    *out = c;
    return 0;
}

// Step 6. Callers guarantee no submission is running (uGDS joins its lanes
// and drains the handle first). Also used on half-built controllers.
void ugds_capsule_disconnect(ugds_capsule_ctrl* c) {
    if (!c) return;
    if (c->ka.joinable()) {
        {
            std::lock_guard<std::mutex> g(c->ka_m);
            c->ka_stop = true;
        }
        c->ka_cv.notify_all();
        c->ka.join();
    }
    // Orderly shutdown so the target frees this controller now instead of
    // waiting for KATO to expire.
    if (c->cntlid && !c->dead.load())
        prop_set(c, kRegCc, (uint32_t)(kCcEnable | kCcIoSqes | kCcIoCqes | kCcShutdownNormal));
    for (auto it = c->qs.rbegin(); it != c->qs.rend(); ++it) queue_close(it->get());
    for (auto& m : c->mrs) ibv_dereg_mr(m.mr);
    if (c->admin_mr) ibv_dereg_mr(c->admin_mr);
    free(c->admin_buf);
    // The PD must go before the last cm_id: librdmacm closes the device
    // context when its final id is destroyed.
    if (c->pd) ibv_dealloc_pd(c->pd);
    for (auto& q : c->qs) {
        if (q->id) rdma_destroy_id(q->id);
        if (q->ch) rdma_destroy_event_channel(q->ch);
    }
    delete c;
}

uint64_t ugds_capsule_capacity(const ugds_capsule_ctrl* c) { return c->capacity; }

uint32_t ugds_capsule_block_size(const ugds_capsule_ctrl* c) { return 1u << c->lba_shift; }

// The allocation's own context is pushed for the two range calls, so this
// works from threads whose current context is something else or nothing.
// The dma-buf fd is only needed while registering; the MR keeps its own
// reference to the export.
int ugds_capsule_register(ugds_capsule_ctrl* c, const void* gpu, size_t size) {
    cuInit(0);
    unsigned long long id = buffer_id_of(gpu);
    if (!id) return -EFAULT;
    if (find_mr(c, (uintptr_t)gpu, size, id)) return 0;
    CUcontext ctx = nullptr;
    if (cuPointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)gpu) != CUDA_SUCCESS ||
        !ctx)
        return -EINVAL;
    cuCtxPushCurrent(ctx);
    CUdeviceptr base = 0;
    size_t len = 0;
    int fd = -1;
    int r = 0;
    if (cuMemGetAddressRange(&base, &len, (CUdeviceptr)gpu) != CUDA_SUCCESS)
        r = -EINVAL;
    else if (cuMemGetHandleForAddressRange(&fd, base, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0) !=
             CUDA_SUCCESS)
        r = -ENOTSUP;
    CUcontext popped;
    cuCtxPopCurrent(&popped);
    if (r) return r;
    if ((uintptr_t)gpu + size > base + len) {
        close(fd);
        return -EINVAL;
    }

    std::lock_guard<std::mutex> g(c->mr_m);
    for (auto& m : c->mrs)
        if (m.base == base && m.buffer_id == id) {
            close(fd);
            return 0;
        }
    // The MR's iova is the GPU virtual address, so SGLs carry plain device
    // pointers.
    ibv_mr* mr = ibv_reg_dmabuf_mr(c->pd, 0, len, base, fd,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                                       IBV_ACCESS_REMOTE_WRITE);
    int err = errno;
    close(fd);
    if (!mr) return -err;
    c->mrs.push_back({(uintptr_t)base, len, id, mr});
    return 0;
}

void ugds_capsule_unregister(ugds_capsule_ctrl* c, const void* gpu) {
    std::lock_guard<std::mutex> g(c->mr_m);
    for (auto it = c->mrs.begin(); it != c->mrs.end(); ++it)
        if ((uintptr_t)gpu >= it->base && (uintptr_t)gpu < it->base + it->len) {
            ibv_dereg_mr(it->mr);
            c->mrs.erase(it);
            return;
        }
}

// 1. Validate every op and look up (or create) its GPU registration;
//    a bad op gets its own error and the rest still run.
// 2. Take one I/O queue for the whole call.
// 3. Choose the piece size from the total bytes and the command budget.
// 4. Issue pieces in rounds of up to queue_depth commands; owner[] maps
//    each command back to its op. A failed piece fails its op; a
//    transport failure fails every op not yet finished.
int ugds_capsule_submit(ugds_capsule_ctrl* c, ugds_capsule_op* ops, int n) {
    const uint64_t blk = 1ull << c->lba_shift;
    ibv_mr* mrs[kMaxDepth];
    std::vector<ibv_mr*> big;
    ibv_mr** mr = n <= kMaxDepth ? mrs : (big.resize(n), big.data());
    for (int i = 0; i < n; ++i) {
        ugds_capsule_op& o = ops[i];
        o.result = 0;
        if (o.off < 0 || o.size == 0 || ((uint64_t)o.off | o.size) & (blk - 1) ||
            (uint64_t)o.off + o.size > c->capacity)
            o.result = -EINVAL;
        else if (int r = get_mr(c, o.gpu, o.size, &mr[i]))
            o.result = r;
    }
    if (c->dead.load(std::memory_order_acquire)) {
        for (int i = 0; i < n; ++i)
            if (!ops[i].result) ops[i].result = -EIO;
        return -EIO;
    }

    Queue* q = take_queue(c);
    std::lock_guard<std::mutex> g(q->m, std::adopt_lock);
    int active = c->active.fetch_add(1, std::memory_order_relaxed) + 1;
    struct Leave {
        std::atomic<int>& a;
        ~Leave() { a.fetch_sub(1, std::memory_order_relaxed); }
    } leave{c->active};

    // Example: one 128 KiB read alone -> budget 16 -> 16 KiB pieces x 8.
    // 32 concurrent 128 KiB reads -> budget 2 -> 64 KiB pieces x 2.
    size_t total = 0;
    for (int j = 0; j < n; ++j)
        if (!ops[j].result) total += ops[j].size;
    size_t budget = (size_t)std::clamp(kInflightBudget / active, 1, q->depth);
    size_t piece = kMinPiece;
    while (piece < kMaxPiece && piece * budget < total) piece <<= 1;
    piece = std::max<size_t>(piece, blk);

    Sqe cmds[kMaxDepth];
    Cqe cqes[kMaxDepth];
    int owner[kMaxDepth];
    int i = 0;
    size_t done = 0;  // bytes of ops[i] already issued
    while (i < n) {
        int k = 0;
        while (k < q->depth && i < n) {
            ugds_capsule_op& o = ops[i];
            if (o.result < 0) {
                ++i;
                done = 0;
                continue;
            }
            size_t len = std::min(piece, o.size - done);
            Sqe& s = cmds[k];
            s = make_cmd(o.is_write ? kOpWrite : kOpRead);
            put32(s, 4, c->t.nsid);
            set_sgl(s, (uint64_t)o.gpu + done, (uint32_t)len, mr[i]->rkey);
            put64(s, 40, ((uint64_t)o.off + done) >> c->lba_shift);
            put32(s, 48, (uint32_t)(len >> c->lba_shift) - 1);
            owner[k++] = i;
            done += len;
            if (done == o.size) {
                ++i;
                done = 0;
            }
        }
        if (k == 0) break;
        if (int r = submit(c, q, cmds, k, cqes, kIoTimeoutMs)) {
            for (int j = 0; j < n; ++j)
                if (!ops[j].result) ops[j].result = r;
            return r;
        }
        for (int j = 0; j < k; ++j)
            if (cqe_status(cqes[j]) && !ops[owner[j]].result) {
                fprintf(stderr, "uGDS capsule: %s status 0x%x\n",
                        ops[owner[j]].is_write ? "write" : "read", cqe_status(cqes[j]));
                ops[owner[j]].result = -EIO;
            }
    }
    for (int j = 0; j < n; ++j)
        if (!ops[j].result) ops[j].result = (ssize_t)ops[j].size;
    return 0;
}

ssize_t ugds_capsule_io(ugds_capsule_ctrl* c, void* gpu, size_t size, off_t off, bool is_write) {
    ugds_capsule_op o{gpu, size, off, is_write, 0};
    ugds_capsule_submit(c, &o, 1);
    return o.result;
}

bool ugds_capsule_alive(const ugds_capsule_ctrl* c) {
    return !c->dead.load(std::memory_order_acquire);
}

int ugds_capsule_queue_depth(const ugds_capsule_ctrl* c) { return c->io_depth; }

int ugds_capsule_nr_queues(const ugds_capsule_ctrl* c) { return (int)c->qs.size() - 1; }
