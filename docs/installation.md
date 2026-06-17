# Installation

## Prerequisites

| Dependency | Version | Notes |
|-----------|---------|-------|
| Linux kernel headers | matching running kernel | tested on 6.8 |
| NVIDIA GPU driver (open kernel modules) | >= 550.x | must use open-source kernel modules; kernel source required (`/usr/src/nvidia-*`) |
| CUDA Toolkit | >= 12.x | needs `nvcc` and `cudart` |
| CMake | >= 3.18 | |
| GCC | C++17 support | |

## Step 1: Build the Kernel Module

The kernel module (`ugds_drv.ko`) provides PCI BAR mapping and GPU page pinning for user-space NVMe access.

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

```bash
mkdir build && cd build
cmake ..
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
    $t -f /dev/ugds_drv0 -d 0
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

