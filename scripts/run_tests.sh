#!/bin/bash
#
# Run UGDS functional and performance tests, plus GDS comparison benchmarks.
# Uses a single NVMe device, switching between modes via env_switch.sh.
#
# Usage:
#   ./scripts/run_tests.sh [functional|perf|compare|all]
#
# Environment variables:
#   PCI_SLOT      PCI slot of the NVMe device  (default: auto-detect ugds_drv device)
#   MOUNT_POINT   Mount point for GDS             (default: /mnt/ugds_test)
#   GPU_ID        GPU device index              (default: 0)
#   BUILD_DIR     Build output directory        (default: ./build)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-${REPO_DIR}/build}"
ENV_SWITCH="${SCRIPT_DIR}/env_switch.sh"

GPU_ID="${GPU_ID:-0}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/ugds_test}"

PASSED=0
FAILED=0
SKIPPED=0

G='\033[0;32m'
R='\033[0;31m'
Y='\033[0;33m'
N='\033[0m'

# ── Auto-detect PCI slot ─────────────────────────────────────────────

detect_pci_slot() {
    if [ -n "${PCI_SLOT:-}" ]; then
        echo "$PCI_SLOT"
        return
    fi
    # Find device bound to ugds_drv
    for dev in /sys/bus/pci/devices/*/; do
        local cls drv
        cls=$(cat "$dev/class" 2>/dev/null || echo "")
        [ "$cls" = "0x010802" ] || continue
        drv=$(basename "$(readlink "$dev/driver" 2>/dev/null)" 2>/dev/null || echo "none")
        if [ "$drv" = "ugds_drv" ]; then
            basename "$dev"
            return
        fi
    done
    echo ""
}

# ── Find UGDS device node ────────────────────────────────────────────

find_ugds_dev() {
    for d in /dev/ugds_drv*; do
        [ -c "$d" ] && echo "$d" && return
    done
    echo ""
}

# ── Ensure UGDS mode ─────────────────────────────────────────────────

ensure_ugds() {
    local slot="$1"
    local drv
    drv=$(basename "$(readlink "/sys/bus/pci/devices/$slot/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [ "$drv" != "ugds_drv" ]; then
        echo ":: Switching $slot to UGDS mode..." >&2
        "$ENV_SWITCH" ugds "$slot" >&2
    fi
    sleep 0.5
    local dev
    dev=$(find_ugds_dev)
    [ -n "$dev" ] || { echo "ERROR: no ugds_drv device after switch" >&2; return 1; }
    echo "$dev"
}

# ── Ensure GDS mode ──────────────────────────────────────────────────

ensure_gds() {
    local slot="$1"
    local drv
    drv=$(basename "$(readlink "/sys/bus/pci/devices/$slot/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [ "$drv" != "nvme" ]; then
        echo ":: Switching $slot to GDS mode..." >&2
        "$ENV_SWITCH" gds "$slot" "$MOUNT_POINT" >&2
    else
        # Ensure mounted
        if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
            local nvme_dev
            for nd in /sys/bus/pci/devices/$slot/nvme/*/; do
                nvme_dev="/dev/$(basename "$nd")n1"
                break
            done
            if [ -n "${nvme_dev:-}" ] && [ -b "$nvme_dev" ]; then
                sudo mkdir -p "$MOUNT_POINT"
                sudo blkid "$nvme_dev" > /dev/null 2>&1 || sudo mkfs.ext4 -F "$nvme_dev" >&2
                sudo mount -o data=ordered "$nvme_dev" "$MOUNT_POINT"
            fi
        fi
    fi
    sleep 1
    echo "$MOUNT_POINT/test_data"
}

# ── Test runner ───────────────────────────────────────────────────────

run_test() {
    local name="$1"
    local bin="$2"
    shift 2
    printf "  %-40s " "$name"
    if [ ! -x "$bin" ]; then
        printf "${Y}SKIP${N} (not built)\n"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    local output
    if output=$("$bin" "$@" 2>&1); then
        printf "${G}PASS${N}\n"
        PASSED=$((PASSED + 1))
    else
        printf "${R}FAIL${N}\n"
        echo "$output" | tail -5 | sed 's/^/    /'
        FAILED=$((FAILED + 1))
    fi
}

# ── Functional tests (UGDS mode) ─────────────────────────────────────

run_functional() {
    local slot="$1"
    echo "=== Functional Tests (UGDS) ==="

    local ugds_dev
    ugds_dev=$(ensure_ugds "$slot")
    echo "  Device: $ugds_dev  GPU: $GPU_ID"
    echo ""

    local tests=(
        test_driver_lifecycle
        test_handle_register
        test_buf_register
        test_read_write_basic
        test_read_write_large
        test_read_write_unregistered
        test_alignment_errors
        test_multi_offset
        test_concurrent_qps
    )

    for t in "${tests[@]}"; do
        run_test "$t" "$BUILD_DIR/$t" "$ugds_dev" "$GPU_ID"
    done
    echo ""
}

# ── UGDS perf (UGDS mode) ────────────────────────────────────────────

run_perf_ugds() {
    local slot="$1"
    echo "=== UGDS Performance ==="

    local ugds_dev
    ugds_dev=$(ensure_ugds "$slot")

    if [ ! -x "$BUILD_DIR/bench_ugds" ]; then
        echo "  SKIP (bench_ugds not built)"
        echo ""
        return
    fi

    echo "  Device: $ugds_dev  GPU: $GPU_ID"
    echo ""

    for sz in 4K 128K 1M; do
        echo "[UGDS - ${sz} seq read]"
        "$BUILD_DIR/bench_ugds" -f "$ugds_dev" -l 128M -s "$sz" -t 1 -d "$GPU_ID" -m read
    done
    echo ""
}

# ── GDS perf (GDS mode, same device) ─────────────────────────────────

run_perf_gds() {
    local slot="$1"
    echo "=== GDS Performance ==="

    if [ ! -x "$BUILD_DIR/bench_gds" ]; then
        echo "  SKIP (bench_gds not built, pass -DCUFILE_LIB=... to cmake)"
        echo ""
        return
    fi

    local gds_file
    gds_file=$(ensure_gds "$slot")

    # Create test file
    if [ ! -f "$gds_file" ]; then
        echo "  Creating test file: $gds_file (256MB)"
        dd if=/dev/urandom of="$gds_file" bs=1M count=256 oflag=direct 2>/dev/null
    fi

    echo "  File: $gds_file  GPU: $GPU_ID"
    echo ""

    for sz in 4K 128K 1M; do
        echo "[GDS - ${sz} seq read]"
        "$BUILD_DIR/bench_gds" -f "$gds_file" -l 128M -s "$sz" -t 1 -d "$GPU_ID" -m read
    done
    echo ""
}

# ── Compare: run UGDS then GDS on the same device ────────────────────

run_compare() {
    local slot="$1"
    echo "============================================"
    echo "  Comparison: UGDS vs GDS (same device $slot)"
    echo "============================================"
    echo ""

    run_perf_ugds "$slot"

    run_perf_gds "$slot"

    # Switch back to UGDS for subsequent operations
    echo ":: Switching back to UGDS mode..."
    "$ENV_SWITCH" ugds "$slot" > /dev/null 2>&1 || true
}

# ── Main ─────────────────────────────────────────────────────────────

mode="${1:-all}"

echo "========================================"
echo "  UserSpace-GDS Test Runner"
echo "========================================"
echo ""

PCI_SLOT_RESOLVED=$(detect_pci_slot)
if [ -z "$PCI_SLOT_RESOLVED" ]; then
    echo "ERROR: Cannot detect NVMe device. Set PCI_SLOT=<slot> environment variable."
    exit 1
fi
echo "  Target device: $PCI_SLOT_RESOLVED"
echo ""

case "$mode" in
    functional)
        run_functional "$PCI_SLOT_RESOLVED"
        ;;
    perf)
        run_perf_ugds "$PCI_SLOT_RESOLVED"
        ;;
    compare)
        run_compare "$PCI_SLOT_RESOLVED"
        ;;
    all)
        run_functional "$PCI_SLOT_RESOLVED"
        run_compare "$PCI_SLOT_RESOLVED"
        ;;
    *)
        echo "Usage: $0 [functional|perf|compare|all]"
        exit 1
        ;;
esac

echo "========================================"
printf "  Results: ${G}%d passed${N}, ${R}%d failed${N}, ${Y}%d skipped${N}\n" "$PASSED" "$FAILED" "$SKIPPED"
echo "========================================"

[ "$FAILED" -eq 0 ] || exit 1
