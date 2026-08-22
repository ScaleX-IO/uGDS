#!/usr/bin/env bash
set -Eeuo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UGDS_REPO="${UGDS_REPO:-$(cd -- "$BENCH_DIR/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$UGDS_REPO")}"
LMCACHE_REPO="${LMCACHE_REPO:-$WORKSPACE_DIR/LMCache}"
VLLM_REPO="${VLLM_REPO:-$WORKSPACE_DIR/vllm}"
UGDS_PCI_SLOT="${UGDS_PCI_SLOT:-}"
UGDS_DEVICE="${UGDS_DEVICE:-}"
SKIP_SETUP="${SKIP_SETUP:-0}"
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-0.6B}"
MODEL="${MODEL:-$BENCH_DIR/models/Qwen3-0.6B}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

normalize_slot() {
    local slot="$1"
    [[ "$slot" == ????:* ]] || slot="0000:$slot"
    printf '%s\n' "$slot"
}

get_driver() {
    local link="/sys/bus/pci/devices/$1/driver"
    if [[ -L "$link" ]]; then
        basename "$(readlink -f "$link")"
    else
        printf 'none\n'
    fi
}

list_ugds_nodes() {
    local node
    for node in /dev/ugds_drv*; do
        [[ -c "$node" ]] && printf '%s\n' "$node"
    done
}

check_target_not_mounted() {
    local pci_dir="/sys/bus/pci/devices/$1"
    local controller namespace block_device mounts
    for controller in "$pci_dir"/nvme/nvme*; do
        [[ -d "$controller" ]] || continue
        for namespace in "$controller"/nvme*n*; do
            [[ -e "$namespace" ]] || continue
            block_device="/dev/$(basename "$namespace")"
            [[ -b "$block_device" ]] || continue
            mounts="$(lsblk -nrpo NAME,MOUNTPOINT "$block_device" | awk 'NF > 1')"
            if [[ -n "$mounts" ]]; then
                printf '%s\n' "$mounts" >&2
                die "$block_device or one of its partitions is mounted"
            fi
        done
    done
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    if [[ -x "$BENCH_DIR/.tools/uv" ]]; then
        printf '%s\n' "$BENCH_DIR/.tools/uv"
        return
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
    printf '==> Installing uv locally under %s/.tools\n' "$BENCH_DIR" >&2
    local installer
    installer="$(mktemp)"
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    UV_UNMANAGED_INSTALL="$BENCH_DIR/.tools" sh "$installer"
    rm -f "$installer"
    printf '%s\n' "$BENCH_DIR/.tools/uv"
}

select_new_ugds_node() {
    local before="$1"
    local node
    while IFS= read -r node; do
        [[ -n "$node" ]] || continue
        if ! grep -Fxq "$node" <<<"$before"; then
            printf '%s\n' "$node"
            return
        fi
    done < <(list_ugds_nodes)
}

[[ "${I_UNDERSTAND_UGDS_ERASES_DEVICE:-0}" == "1" ]] || die \
    "set I_UNDERSTAND_UGDS_ERASES_DEVICE=1 after confirming the target SSD may be overwritten"
[[ -n "$UGDS_PCI_SLOT" ]] || die \
    "set UGDS_PCI_SLOT to the dedicated NVMe controller (example: 0000:b8:00.0)"
UGDS_PCI_SLOT="$(normalize_slot "$UGDS_PCI_SLOT")"
PCI_DIR="/sys/bus/pci/devices/$UGDS_PCI_SLOT"
[[ -d "$PCI_DIR" ]] || die "PCI device not found: $UGDS_PCI_SLOT"
[[ "$(<"$PCI_DIR/class")" == "0x010802" ]] || die \
    "$UGDS_PCI_SLOT is not an NVMe controller"

printf '==> Target NVMe\n'
printf 'PCI slot: %s\n' "$UGDS_PCI_SLOT"
printf 'PCI ID: %s:%s\n' "$(<"$PCI_DIR/vendor")" "$(<"$PCI_DIR/device")"
printf 'Current driver: %s\n' "$(get_driver "$UGDS_PCI_SLOT")"
check_target_not_mounted "$UGDS_PCI_SLOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$BENCH_DIR/.uv-cache}"
if [[ "$SKIP_SETUP" != "1" ]]; then
    UV_BIN="${UV_BIN:-$(ensure_uv)}"
    printf '==> Building and installing the software stack\n'
    UV_BIN="$UV_BIN" BUILD_UGDS_DRIVER=1 \
        UGDS_REPO="$UGDS_REPO" LMCACHE_REPO="$LMCACHE_REPO" \
        VLLM_REPO="$VLLM_REPO" EXISTING_PYTHON="${EXISTING_PYTHON:-}" \
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
        "$BENCH_DIR/setup_env.sh"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -n "${EXISTING_PYTHON:-}" ]]; then
    PYTHON_BIN="$EXISTING_PYTHON"
elif [[ -f "$BENCH_DIR/.python-bin" ]]; then
    PYTHON_BIN="$(<"$BENCH_DIR/.python-bin")"
else
    PYTHON_BIN="$BENCH_DIR/.venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || die "Python environment is not ready: $PYTHON_BIN"

current_driver="$(get_driver "$UGDS_PCI_SLOT")"
if [[ "$current_driver" != "ugds_drv" ]]; then
    before_nodes="$(list_ugds_nodes || true)"
    printf '==> Binding %s to ugds_drv\n' "$UGDS_PCI_SLOT"
    "$UGDS_REPO/scripts/env_switch.sh" ugds "$UGDS_PCI_SLOT"
    if [[ -z "$UGDS_DEVICE" ]]; then
        UGDS_DEVICE="$(select_new_ugds_node "$before_nodes")"
    fi
fi

if [[ -z "$UGDS_DEVICE" ]]; then
    mapfile -t nodes < <(list_ugds_nodes)
    if [[ "${#nodes[@]}" == "1" ]]; then
        UGDS_DEVICE="${nodes[0]}"
    else
        die "multiple uGDS nodes exist; set UGDS_DEVICE for PCI $UGDS_PCI_SLOT"
    fi
fi
[[ -c "$UGDS_DEVICE" ]] || die "uGDS character device not found: $UGDS_DEVICE"
sudo chown "$(id -u):$(id -g)" "$UGDS_DEVICE"
sudo chmod 600 "$UGDS_DEVICE"
printf 'uGDS device: %s\n' "$UGDS_DEVICE"

if [[ ! -s "$MODEL/config.json" ]]; then
    printf '==> Downloading %s to %s\n' "$HF_MODEL_ID" "$MODEL"
    "$PYTHON_BIN" - "$HF_MODEL_ID" "$MODEL" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
fi

if [[ -n "${EXISTING_PYTHON:-}" ]]; then
    USE_LOCAL_VLLM_SOURCE="${USE_LOCAL_VLLM_SOURCE:-0}"
else
    USE_LOCAL_VLLM_SOURCE="${USE_LOCAL_VLLM_SOURCE:-1}"
fi

printf '==> Running vLLM + LMCache + uGDS end-to-end benchmark\n'
env PYTHON_BIN="$PYTHON_BIN" MODEL="$MODEL" UGDS_DEVICE="$UGDS_DEVICE" \
    UGDS_REPO="$UGDS_REPO" LMCACHE_REPO="$LMCACHE_REPO" \
    VLLM_REPO="$VLLM_REPO" USE_LOCAL_VLLM_SOURCE="$USE_LOCAL_VLLM_SOURCE" \
    UGDS_PCI_SLOT="$UGDS_PCI_SLOT" I_UNDERSTAND_UGDS_ERASES_DEVICE=1 \
    "$BENCH_DIR/run_e2e_bench.sh"
