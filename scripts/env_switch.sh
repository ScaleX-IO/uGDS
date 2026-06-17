#!/bin/bash
#
# Environment switching tool for UGDS / GDS testing.
#
# Usage:
#   ./scripts/env_switch.sh status                    Show current state
#   ./scripts/env_switch.sh ugds  <pci_slot>          Switch device to UGDS mode
#   ./scripts/env_switch.sh gds   <pci_slot> [mount]  Switch device to GDS mode
#
# Examples:
#   ./scripts/env_switch.sh status
#   ./scripts/env_switch.sh ugds 0000:b8:00.0
#   ./scripts/env_switch.sh gds  0000:27:00.0 /mnt/nvme0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

UGDS_DRV_MODULE="${REPO_DIR}/drv/ugds_drv.ko"
NVIDIA_FS_MODULE="nvidia-fs"

SYSFS_PCI="/sys/bus/pci/devices"
SYSFS_DRV="/sys/bus/pci/drivers"
NVME_CLASS="0x010802"

# ── Helpers ──────────────────────────────────────────────────────────

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ":: $*"; }

normalize_slot() {
    local slot="$1"
    [[ "$slot" == ????:* ]] || slot="0000:$slot"
    echo "$slot"
}

get_driver() {
    local slot="$1"
    local drv_link="$SYSFS_PCI/$slot/driver"
    if [ -L "$drv_link" ]; then
        basename "$(readlink "$drv_link")"
    else
        echo "none"
    fi
}

get_nvme_dev() {
    local slot="$1"
    local nvme_dir="$SYSFS_PCI/$slot/nvme"
    if [ -d "$nvme_dir" ]; then
        local ctrl
        ctrl=$(ls "$nvme_dir" 2>/dev/null | head -1)
        [ -n "$ctrl" ] && echo "/dev/${ctrl}n1" && return
    fi
    echo ""
}

unbind_device() {
    local slot="$1"
    local drv
    drv=$(get_driver "$slot")
    if [ "$drv" = "none" ]; then
        info "$slot already unbound"
        return
    fi
    info "Unbinding $slot from $drv"
    echo -n "$slot" | sudo tee "$SYSFS_DRV/$drv/unbind" > /dev/null
    sleep 0.5
}

bind_device() {
    local slot="$1"
    local target_drv="$2"
    local cur_drv
    cur_drv=$(get_driver "$slot")
    if [ "$cur_drv" = "$target_drv" ]; then
        info "$slot already bound to $target_drv"
        return
    fi
    if [ "$cur_drv" != "none" ]; then
        unbind_device "$slot"
    fi
    info "Binding $slot to $target_drv"
    echo -n "$slot" | sudo tee "$SYSFS_DRV/$target_drv/bind" > /dev/null
    sleep 1
}

list_nvme_devices() {
    for dev in "$SYSFS_PCI"/*/; do
        local slot
        slot=$(basename "$dev")
        local cls
        cls=$(cat "$dev/class" 2>/dev/null || echo "")
        [ "$cls" = "$NVME_CLASS" ] || continue
        local drv
        drv=$(get_driver "$slot")
        local nvme_dev
        nvme_dev=$(get_nvme_dev "$slot")
        local ugds_dev=""
        if [ "$drv" = "ugds_drv" ]; then
            # Find ugds_drv device node
            for d in /dev/ugds_drv*; do
                [ -c "$d" ] && ugds_dev="$d" && break
            done
        fi
        local mount_point=""
        if [ -n "$nvme_dev" ] && [ -b "$nvme_dev" ]; then
            mount_point=$(findmnt -n -o TARGET "$nvme_dev" 2>/dev/null || echo "")
        fi
        printf "  %-14s  driver=%-10s" "$slot" "$drv"
        [ -n "$nvme_dev" ] && printf "  dev=%-14s" "$nvme_dev" || printf "  %-18s" ""
        [ -n "$ugds_dev" ] && printf "  ugds=%s" "$ugds_dev"
        [ -n "$mount_point" ] && printf "  mount=%s" "$mount_point"
        printf "\n"
    done
}

# ── Commands ─────────────────────────────────────────────────────────

cmd_status() {
    echo "=== NVMe Devices ==="
    list_nvme_devices
    echo ""
    echo "=== Relevant Modules ==="
    lsmod | grep -E 'ugds_drv|nvidia_fs|^nvme ' | while read -r line; do
        echo "  $line"
    done
    echo ""
    echo "=== UGDS ==="
    ls /dev/ugds_drv* 2>/dev/null | while read -r d; do echo "  $d"; done || echo "  (no devices)"
    echo ""
    echo "=== GDS ==="
    if [ -f /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcufile.so ]; then
        echo "  libcufile: available"
    else
        echo "  libcufile: NOT FOUND"
    fi
    if lsmod | grep -q nvidia_fs; then
        echo "  nvidia-fs: loaded"
    else
        echo "  nvidia-fs: not loaded"
    fi
}

cmd_ugds() {
    local slot
    slot=$(normalize_slot "${1:?'PCI slot required'}")
    [ -d "$SYSFS_PCI/$slot" ] || die "Device $slot not found"

    local cur_drv
    cur_drv=$(get_driver "$slot")

    # If mounted, unmount first
    if [ "$cur_drv" = "nvme" ]; then
        local nvme_dev
        nvme_dev=$(get_nvme_dev "$slot")
        if [ -n "$nvme_dev" ]; then
            local mnt
            mnt=$(findmnt -n -o TARGET "$nvme_dev" 2>/dev/null || echo "")
            if [ -n "$mnt" ]; then
                info "Unmounting $nvme_dev from $mnt"
                sudo umount "$mnt"
            fi
        fi
    fi

    # Unbind from current driver
    unbind_device "$slot"

    # Load ugds_drv if not loaded
    if ! lsmod | grep -q "^ugds_drv "; then
        [ -f "$UGDS_DRV_MODULE" ] || die "ugds_drv.ko not found at $UGDS_DRV_MODULE (run 'make' in drv/)"
        info "Loading ugds_drv module"
        sudo insmod "$UGDS_DRV_MODULE" max_num_ctrls=64 2>/dev/null || true
        sleep 1
    fi

    # Bind to ugds_drv
    bind_device "$slot" "ugds_drv"

    info "Done. UGDS device:"
    ls /dev/ugds_drv* 2>/dev/null | head -1
}

cmd_gds() {
    local slot
    slot=$(normalize_slot "${1:?'PCI slot required'}")
    local mount_point="${2:-}"
    [ -d "$SYSFS_PCI/$slot" ] || die "Device $slot not found"

    local cur_drv
    cur_drv=$(get_driver "$slot")

    # If using ugds_drv, unbind
    if [ "$cur_drv" = "ugds_drv" ]; then
        unbind_device "$slot"
    fi

    # Bind to nvme
    bind_device "$slot" "nvme"
    sleep 1

    # Load nvidia-fs if not loaded
    if ! lsmod | grep -q "^nvidia_fs "; then
        info "Loading nvidia-fs module"
        sudo modprobe "$NVIDIA_FS_MODULE" 2>/dev/null || info "WARNING: nvidia-fs failed to load"
    fi

    # Find NVMe device
    local nvme_dev
    nvme_dev=$(get_nvme_dev "$slot")
    [ -n "$nvme_dev" ] || die "Could not find NVMe block device for $slot"
    info "NVMe device: $nvme_dev"

    # Mount if requested
    if [ -n "$mount_point" ]; then
        if findmnt -n "$nvme_dev" > /dev/null 2>&1; then
            info "$nvme_dev already mounted"
        else
            sudo mkdir -p "$mount_point"
            # Check if filesystem exists
            if ! sudo blkid "$nvme_dev" > /dev/null 2>&1; then
                info "No filesystem on $nvme_dev, creating ext4"
                sudo mkfs.ext4 -F "$nvme_dev"
            fi
            info "Mounting $nvme_dev at $mount_point"
            sudo mount -o data=ordered "$nvme_dev" "$mount_point"
        fi
        info "Done. GDS test file: $mount_point/test_data"
    else
        info "Done. NVMe device ready: $nvme_dev (mount it for GDS file I/O)"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  status                         Show NVMe devices, drivers, and module state"
    echo "  ugds    <pci_slot>             Switch device to UGDS mode (ugds_drv)"
    echo "  gds     <pci_slot> [mount]     Switch device to GDS mode (nvme + nvidia-fs)"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 ugds b8:00.0"
    echo "  $0 gds  27:00.0 /mnt/nvme0"
}

case "${1:-}" in
    status)  cmd_status ;;
    ugds)    cmd_ugds "${2:-}" ;;
    gds)     cmd_gds "${2:-}" "${3:-}" ;;
    -h|--help|"") usage ;;
    *) die "Unknown command: $1" ;;
esac
