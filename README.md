<p align="center">
  <img src="assets/logo.png" alt="uGDS Logo" width="500">
</p>

<p align="center">
  <b>User-Space GPU Direct Storage Development Kit</b>
</p>

<p align="center">
  uGDS is the first production-oriented user-space GPU Direct Storage library where the CPU constructs NVMe commands and the SSD DMAs data directly to/from GPU memory over PCIe, bypassing the kernel NVMe driver. With a GDS-compatible API, uGDS achieves up to 5.3x higher bandwidth and 108x lower 4KB latency (5.2μs vs 561μs) compared to NVIDIA GDS.
</p>

---

## Key Features

- **User-space IO stack** — bypasses the kernel NVMe driver and filesystem entirely; CPU builds NVMe commands and polls completions in user space
- **cuFile API compatible** — existing GDS applications work with minimal changes (relink to `libugds.so`, change `cuFile` prefix to `uGDS`)
- **Multi-vendor GPU support** — NVIDIA CUDA and AMD Infinity Storage (HIP/ROCm) backends; both can be enabled simultaneously for mixed-GPU systems
- **Fully open-source** — BSD 3-Clause licensed; no proprietary runtime dependencies beyond the GPU driver (NVIDIA driver or AMD ROCm runtime)
- **High performance** — busy-poll CQ completion with `_mm_pause()`, multi-queue round-robin IO, achieving up to 2.7x read and 28x write bandwidth over NVIDIA GDS

## Architecture

```
┌──────────────┐
│  Application │    uGDSRead / uGDSWrite
└──────┬───────┘
       │
┌──────▼───────┐
│  libugds.so  │    NVMe command construction + SQ/CQ management
└──────┬───────┘
       │          DMA: SSD ←→ GPU memory (P2P over PCIe)
┌──────▼───────┐
│  ugds_drv.ko │    PCI BAR mapping + GPU page pinning
└──────┬───────┘
       │
┌──────▼───────┐
│   NVMe SSD   │
└──────────────┘
```

No kernel NVMe driver, no page cache — the CPU only touches doorbell registers and completion queue entries.

## Performance

16-thread sequential read bandwidth on A100-40GB + Samsung 990 PRO (PCIe Gen4 x4):

![Sequential Read: GDS vs uGDS](assets/ugds_vs_gds_16t_read.png)

![Sequential Write: GDS vs uGDS](assets/ugds_vs_gds_16t_write.png)

uGDS bypasses the kernel NVMe driver, achieving up to **2.7x** higher read bandwidth and **28x** higher write bandwidth than NVIDIA GDS at small IO sizes.

## Quick Start

```cpp
#include <ugds.h>
#include <cuda_runtime.h>

// 1. Initialize
uGDSDriverOpen();

// 2. Register device handle
int fd = open("/dev/ugds_drv0", O_RDWR);
uGDSDescr_t desc = {.type = UGDS_HANDLE_TYPE_OPAQUE_FD, .handle.fd = fd};
uGDSHandle_t fh;
uGDSHandleRegister(&fh, &desc);

// 3. Allocate and register GPU buffer
void* gpu_buf;
cudaMalloc(&gpu_buf, 4096);
uGDSBufRegister(gpu_buf, 4096, 0);

// 4. Direct GPU ←→ SSD transfer
uGDSWrite(fh, gpu_buf, 4096, /*file_offset=*/0, /*buf_offset=*/0);
uGDSRead(fh, gpu_buf, 4096, /*file_offset=*/0, /*buf_offset=*/0);

// 5. Cleanup
uGDSBufDeregister(gpu_buf);
uGDSHandleDeregister(fh);
uGDSDriverClose();
```

See [examples/01_basic_read_write.cu](examples/01_basic_read_write.cu) for a complete working example.

For build instructions, environment setup, and driver management, see the **[Installation Guide](docs/installation.md)**.

### AMD Infinity Storage (HIP/ROCm) Backend

To build with the AMD HIP backend instead of CUDA:

```bash
# Kernel module (HIP-only if NVIDIA source is present, add BUILD_CUDA=0)
cd drv && make BUILD_HIP=1 BUILD_CUDA=0

# Userspace library
mkdir build && cd build
cmake .. -DUGDS_BACKEND_HIP=ON -DUGDS_BACKEND_CUDA=OFF
make -j$(nproc)
```

Requires ROCm 5.6+, `CONFIG_HSA_AMD_P2P=y`, and Large BAR enabled. See the [Installation Guide](docs/installation.md) for details.

## Testing

```bash
# Run functional tests only
scripts/run_tests.sh functional

# Run uGDS performance benchmark
scripts/run_tests.sh perf

# Run uGDS vs GDS comparison (auto-switches driver mode)
scripts/run_tests.sh compare

# Run all (functional + comparison)
scripts/run_tests.sh all
```

## API Coverage

| API | Status | Notes |
|-----|:------:|-------|
| `uGDSDriverOpen / Close` | ✅ | |
| `uGDSHandleRegister / Deregister` | ✅ | Block device fd (no filesystem) |
| `uGDSBufRegister / Deregister` | ✅ | GPU memory only |
| `uGDSRead / Write` | ✅ | Synchronous, block-aligned |
| `uGDSBatchIOSetUp / Submit / GetStatus / Destroy` | ✅ | Submit/poll separation, up to 128 IOs per batch |
| `uGDSReadAsync / WriteAsync` | ✅ | CUDA stream integration, late-binding pointers |
| `uGDSStreamRegister / Deregister` | ✅ | Optional (no-op, uGDS has no bounce buffer) |

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core synchronous API + test suite | ✅ |
| 1.5 | Unified multi-backend (NVIDIA CUDA + AMD HIP/ROCm) | 🔧 |
| 2 | Batch IO API (multi-command doorbell) | ✅ |
| 3 | Async Stream API (CUDA stream integration) | ✅ |
| 4 | Hugepage support (larger QP depth) | ✅ |
| 5 | SGL support (scatter-gather lists) | 🔜 |
| 6 | Interrupt mode (MSI-X + eventfd) | 🔜 |
| 7 | Multi-SSD support (multi-handle aggregation) | 🔜 |
| 8 | Striping (automatic IO distribution across SSDs) | 🔜 |
| 9 | Filesystem compatibility (POSIX file path support) | 🔜 |
| 10 | LMCache integration (KV cache storage backend) | 🔧 |

## Citation

uGDS originated from the motivation experiments in CoPilotIO. If you find uGDS useful in your research, please cite:

```bibtex
@inproceedings{chen2026copilotio,
  title     = {CoPilotIO: CPU as a Co-pilot for GPU I/O to Free GPU Compute},
  author    = {Guanyi Chen and Qi Chen and Shu Yin and Jian Zhang},
  booktitle = {Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI '26)},
  year      = {2026}
}
```

## References

- [ssd-gpu-dma](https://github.com/enfiskutensykkel/ssd-gpu-dma) — User-space NVMe driver with GPU support
- [BaM](https://github.com/ZaidQureshi/bam) — Big accelerator Memory, GPU-orchestrated NVMe access
- [Phoenix](https://github.com/xPU-IO/phoenix) — GPU Direct Storage Optimization

## Contact

- Guanyi Chen — felixlinker02@gmail.com — [chengy-sysu.github.io](https://chengy-sysu.github.io)
- Jian Zhang — jianz@hkust-gz.edu.cn — [sosp.dev](https://sosp.dev)

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
