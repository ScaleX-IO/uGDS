KERNEL ?= /lib/modules/$(shell uname -r)/build
NVIDIA_DIR ?= $(shell find /usr/src -maxdepth 1 -type d -regextype posix-extended -regex '.*/nvidia-[0-9]+(\.[0-9]+)+' 2>/dev/null | sort -V | tail -1)
export NVIDIA_DIR

# Backend selection (independent, can enable both):
#   BUILD_HIP=1              -> AMD HIP backend (DMA-buf)
#   BUILD_CUDA=1             -> NVIDIA CUDA backend (requires NVIDIA_DIR)
#   BUILD_CUDA=0             -> explicitly disable CUDA (even if NVIDIA_DIR found)
#   BUILD_HIP=1 BUILD_CUDA=1 -> Dual backend
# Default: auto-detect NVIDIA_DIR for CUDA, no HIP unless BUILD_HIP=1
ifeq ($(BUILD_HIP),1)
export BUILD_HIP
endif

ifneq ($(KERNELRELEASE),)
	obj-m := ugds_drv.o
	ugds_drv-objs := pci.o list.o ctrl.o map.o

	# KUnit self-test module (opt-in, HIP-only)
 ifeq ($(BUILD_KUNIT),1)
  ifneq ($(BUILD_HIP),1)
  $(error BUILD_KUNIT requires BUILD_HIP=1 -- SG flatten tests are HIP-only)
  endif
	obj-m += sg_flatten_test.o
	sg_flatten_test-objs := test/sg_flatten_test.o
	ccflags-y += -DUGDS_KUNIT
 endif

	# AMD HIP backend (optional)
 ifeq ($(BUILD_HIP),1)
	ccflags-y += -D_HIP
 endif

	# NVIDIA CUDA backend (optional)
	# BUILD_CUDA: 1 = force enable, 0 = force disable, unset = auto (if NVIDIA_DIR found)
 ifneq ($(BUILD_CUDA),0)
  ifneq ($(strip $(NVIDIA_DIR)),)
	ENABLE_CUDA := 1
	ccflags-y += -D_CUDA -I$(NVIDIA_DIR)/nvidia
	KBUILD_EXTRA_SYMBOLS += $(NVIDIA_DIR)/Module.symvers
  else
  ifeq ($(BUILD_CUDA),1)
	$(error BUILD_CUDA=1 but NVIDIA_DIR not found. Install NVIDIA driver source or set NVIDIA_DIR=<path>)
  endif
  endif
 endif

	# Error if neither backend is selected
 ifndef ENABLE_CUDA
 ifneq ($(BUILD_HIP),1)
	$(error No GPU backend selected. Set BUILD_HIP=1 for AMD HIP, or install NVIDIA driver source for CUDA)
 endif
 endif

	# dmabuf backend available when HIP is enabled, or when CUDA
	# explicitly requests it via HAVE_CUDA_DMABUF=1. The userspace
	# CMake detects this with check_cxx_source_compiles. Pass the same
	# flag to the kernel build: make BUILD_CUDA=1 HAVE_CUDA_DMABUF=1
UGDS_HAVE_DMABUF := 0
ifeq ($(BUILD_HIP),1)
  UGDS_HAVE_DMABUF := 1
endif
ifdef HAVE_CUDA_DMABUF
  ifneq ($(HAVE_CUDA_DMABUF),0)
    UGDS_HAVE_DMABUF := 1
  endif
endif

ifeq ($(UGDS_HAVE_DMABUF),1)
  ccflags-y += -DUGDS_HAVE_DMABUF
endif

else

.PHONY: default reload unload load clean

default:
	$(MAKE) -C $(KERNEL) M=$(CURDIR) NVIDIA_DIR=$(NVIDIA_DIR) modules

clean:
	$(MAKE) -C $(KERNEL) M=$(CURDIR) clean

reload: unload load

unload:
	-rmmod ugds_drv

load:
	insmod ugds_drv.ko max_num_ctrls=64

endif
