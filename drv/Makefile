KERNEL ?= /lib/modules/$(shell uname -r)/build
NVIDIA_DIR ?= $(shell find /usr/src -maxdepth 1 -type d -regextype posix-extended -regex '.*/nvidia-[0-9]+(\.[0-9]+)+' 2>/dev/null | sort -V | tail -1)
export NVIDIA_DIR

ifneq ($(KERNELRELEASE),)
	obj-m := ugds_drv.o
	ugds_drv-objs := pci.o list.o ctrl.o map.o
	ccflags-y += -D_CUDA -I$(NVIDIA_DIR)/nvidia
	KBUILD_EXTRA_SYMBOLS += $(NVIDIA_DIR)/Module.symvers
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
