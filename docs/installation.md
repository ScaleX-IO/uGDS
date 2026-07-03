# Installation

## Prerequisites

### NVIDIA CUDA Backend

| Dependency | Version | Notes |
|-----------|---------|-------|
| Linux kernel headers | matching running kernel | tested on 6.8 |
| NVIDIA GPU driver (open kernel modules) | >= 550.x | must use open-source kernel modules; kernel source required (`/usr/src/nvidia-*`) |
| CUDA Toolkit | >= 12.x | needs `nvcc` and `cudart` |
| CMake | >= 3.18 | |
| GCC | C++17 support | |

### AMD Infinity Storage (HIP/ROCm) Backend

The HIP backend implements AMD Infinity Storage (AIS) using the standard Linux DMA-buf framework. It does not depend on NVIDIA-specific APIs.

| Dependency | Version | Notes |
|-----------|---------|-------|
| Linux kernel headers | >= 5.12 (6.4+ recommended) | `CONFIG_PCI_P2PDMA=y`, `CONFIG_DMABUF_MOVE_NOTIFY=y`. `MODULE_IMPORT_NS("DMA_BUF")` requires >= 6.4; on older kernels, remove or guard the import. |
| ROCm | >= 7.0 | `hsa_amd_portable_export_dmabuf_v2()` with `HSA_AMD_DMABUF_MAPPING_TYPE_PCIE` |
| HIP runtime | >= 7.0 | `hipMalloc`, `hipMemcpy` |
| HSA runtime | >= 7.0 | `libhsa-runtime64.so` |
| amdgpu-dkms | >= 6.14.14 | VRAM P2P DMA pinning requires upstream fix in this version |
| GPU | Large BAR | MI300X, MI250, MI210 (Full BAR exposes VRAM via PCIe) |
| CMake | >= 3.18 (CUDA), >= 3.21 (HIP) | |
| GCC | C++17 support | |

**Kernel requirements (mandatory):**
- `CONFIG_PCI_P2PDMA=y` -- PCIe P2P DMA support (required)
- `CONFIG_DMABUF_MOVE_NOTIFY=y` -- DMA-buf dynamic attach (required)
- AMD HSA P2P support in kernel (`CONFIG_HSA_AMD_P2P=y`, depends on `HSA_AMD && PCI_P2PDMA && DMABUF_MOVE_NOTIFY`)
- IOMMU passthrough recommended: `iommu=pt` or `amd_iommu=off`

**Driver notes:**
The kernel-side code uses standard upstream DMA-buf APIs (`dma_buf_dynamic_attach`, `dma_buf_pin`, `dma_buf_map_attachment`). The userspace side uses `hsa_amd_portable_export_dmabuf_v2()` from the ROCm runtime. No vendor-specific amdgpu extensions are required beyond upstream `CONFIG_HSA_AMD_P2P` support.

**Synchronization contract:**
Buffers registered with `uGDSBufRegister()` must not be modified by the GPU (HIP kernel writes) while NVMe I/O is in flight on the same buffer. Concurrent GPU access during DMA can cause data corruption. This mirrors the NVIDIA GDS requirement.

**Important:** Each backend used at runtime must be enabled in both the kernel module and the userspace library. For example, a CUDA-only kernel module will reject HIP `uGDSBufRegister()` calls. In dual-backend builds, use `uGDSBufRegister(ptr, size, UGDS_REGISTER_DMABUF)` for AMD buffers and `uGDSBufRegister(ptr, size, 0)` for NVIDIA buffers (default).

**Note:** In dual-backend builds, unregistered AMD buffers (on-the-fly I/O without prior `uGDSBufRegister`) are not supported. AMD buffers must be explicitly registered before I/O. NVIDIA buffers support both registered and unregistered paths.

## Step 1: Build the Kernel Module

The kernel module (`ugds_drv.ko`) provides PCI BAR mapping and GPU page pinning for user-space NVMe access.

### NVIDIA CUDA Backend (default)

```bash
cd drv
make
```

The Makefile auto-detects:
- Kernel build directory via `/lib/modules/$(uname -r)/build`
- NVIDIA driver source via `/usr/src/nvidia-*`

If auto-detection fails, specify manually:

```bash
make KERNEL=/path/to/kernel/build NVIDIA_DIR=/usr/src/nvidia-550.54.14
```

### AMD Infinity Storage (HIP) Backend

```bash
cd drv
make BUILD_HIP=1 BUILD_CUDA=0
```

No NVIDIA driver headers needed -- the HIP path uses standard Linux DMA-buf framework. Kernel headers (`/lib/modules/$(uname -r)/build`) are still required.

## Step 2: Load the Kernel Module

```bash
# Unbind NVMe device from kernel driver first
echo "<your-pci-slot>" | sudo tee /sys/bus/pci/drivers/nvme/unbind

# Load uGDS driver
sudo insmod drv/ugds_drv.ko max_num_ctrls=64

# Bind the NVMe device
echo "<your-pci-slot>" | sudo tee /sys/bus/pci/drivers/ugds_drv/bind

# Verify
ls /dev/ugds_drv0
```

Or use the convenience script:

```bash
scripts/env_switch.sh ugds <your-pci-slot>
```

To check the current state of all NVMe devices:

```bash
scripts/env_switch.sh status
```

## Step 3: Build the Library

### NVIDIA CUDA (default)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### AMD HIP/ROCm

```bash
mkdir build && cd build
cmake .. -DUGDS_BACKEND_CUDA=OFF -DUGDS_BACKEND_HIP=ON
make -j$(nproc)
```

Build outputs:

| Target | Description |
|--------|-------------|
| `libugds.so` | Shared library (uGDS API) |
| `bench_ugds` | Performance benchmark tool |
| `test_driver_lifecycle` | Functional test: driver open/close |
| `test_handle_register` | Functional test: handle register/deregister |
| `test_buf_register` | Functional test: buffer register/deregister |
| `test_read_write_basic` | Functional test: 4KB write + read-back |
| `test_read_write_large` | Functional test: 1MB IO (cross-MDTS) |
| `test_read_write_unregistered` | Functional test: on-the-fly DMA path |
| `test_alignment_errors` | Functional test: alignment error handling |
| `test_multi_offset` | Functional test: multi-offset LBA calculation |
| `test_concurrent_qps` | Functional test: 4-thread concurrent IO |

### Optional: Build GDS Comparison Benchmark

To compare uGDS against NVIDIA GDS on the same hardware:

```bash
cmake .. -DCUFILE_LIB=/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so
make -j$(nproc)
```

This builds `bench_gds` alongside `bench_ugds` for direct performance comparison.

## Step 4: Verify

Run the functional test suite:

```bash
for t in build/test_*; do
    echo "=== $(basename $t) ==="
    $t /dev/ugds_drv0 0
done
```

Run a quick performance check:

```bash
# Single-thread 4KB read
./build/bench_ugds -f /dev/ugds_drv0 -l 128M -s 4K -t 1 -d 0 -m read

# 16-thread 64KB read
./build/bench_ugds -f /dev/ugds_drv0 -l 256M -s 64K -t 16 -d 0 -m read
```

## Environment Switching

The `scripts/env_switch.sh` tool manages driver binding for the same NVMe device across different backends:

```bash
# Switch to uGDS (user-space NVMe)
scripts/env_switch.sh ugds <your-pci-slot>

# Switch to GDS (kernel NVMe + nvidia-fs)
scripts/env_switch.sh gds <your-pci-slot> /mnt/nvme_test

# Check status
scripts/env_switch.sh status
```

