#!/usr/bin/env bash
set -Eeuo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UGDS_REPO="${UGDS_REPO:-$(cd -- "$BENCH_DIR/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$UGDS_REPO")}"
LMCACHE_REPO="${LMCACHE_REPO:-$WORKSPACE_DIR/LMCache}"
VLLM_REPO="${VLLM_REPO:-$WORKSPACE_DIR/vllm}"
if [[ -z "${PYTHON_BIN:-}" && -f "$BENCH_DIR/.python-bin" ]]; then
    PYTHON_BIN="$(<"$BENCH_DIR/.python-bin")"
fi
PYTHON_BIN="${PYTHON_BIN:-$BENCH_DIR/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
UGDS_DEVICE="${UGDS_DEVICE:-/dev/ugds_drv0}"
UGDS_LIB_DIR="${UGDS_LIB_DIR:-$UGDS_REPO/build}"
GPU_ID="${GPU_ID:-0}"
L1_SIZE_GB="${L1_SIZE_GB:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PROMPT_REPEATS="${PROMPT_REPEATS:-64}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
USE_LOCAL_VLLM_SOURCE="${USE_LOCAL_VLLM_SOURCE:-1}"
LMCACHE_CONNECTOR_MODULE="${LMCACHE_CONNECTOR_MODULE:-}"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
ARTIFACT_DIR="${ARTIFACT_DIR:-$BENCH_DIR/artifacts/$RUN_ID}"
LMCACHE_PID=""
VLLM_PID=""

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

stop_process_group() {
    local pid="$1"
    local name="$2"
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        printf 'Stopping %s (process group %s)\n' "$name" "$pid"
        kill -TERM -- "-$pid" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 0.25
        done
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi
}

cleanup() {
    local status=$?
    if [[ "$KEEP_SERVERS" == "1" && "$status" == "0" ]]; then
        printf 'KEEP_SERVERS=1: LMCache PID=%s, vLLM PID=%s\n' \
            "$LMCACHE_PID" "$VLLM_PID"
    else
        stop_process_group "$VLLM_PID" "vLLM"
        stop_process_group "$LMCACHE_PID" "LMCache"
    fi
    if [[ "$status" != "0" && -d "$ARTIFACT_DIR" ]]; then
        printf 'Benchmark failed. Logs: %s\n' "$ARTIFACT_DIR" >&2
        [[ -f "$ARTIFACT_DIR/lmcache.log" ]] && \
            tail -n 30 "$ARTIFACT_DIR/lmcache.log" >&2 || true
        [[ -f "$ARTIFACT_DIR/vllm.log" ]] && \
            tail -n 30 "$ARTIFACT_DIR/vllm.log" >&2 || true
    elif [[ "$status" != "0" ]]; then
        printf 'Benchmark failed during preflight; no artifacts were created.\n' >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_url() {
    local url="$1"
    local pid="$2"
    local name="$3"
    local timeout_seconds="$4"
    local deadline=$((SECONDS + timeout_seconds))
    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error --max-time 2 "$url" \
            >/dev/null 2>&1; then
            printf '%s is ready: %s\n' "$name" "$url"
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || die \
            "$name exited before becoming ready; inspect $ARTIFACT_DIR/${name,,}.log"
        sleep 1
    done
    die "$name did not become ready within ${timeout_seconds}s"
}

port_is_busy() {
    local port="$1"
    timeout 1 bash -c "</dev/tcp/127.0.0.1/$port" >/dev/null 2>&1
}

[[ "${I_UNDERSTAND_UGDS_ERASES_DEVICE:-0}" == "1" ]] || die \
    "uGDS overwrites its slab. Re-run with I_UNDERSTAND_UGDS_ERASES_DEVICE=1 after verifying UGDS_DEVICE is a disposable dedicated SSD"

[[ -f "$UGDS_REPO/CMakeLists.txt" ]] || die "invalid uGDS repo: $UGDS_REPO"
[[ -f "$LMCACHE_REPO/pyproject.toml" ]] || die "invalid LMCache repo: $LMCACHE_REPO"
[[ -f "$VLLM_REPO/pyproject.toml" ]] || die "invalid vLLM repo: $VLLM_REPO"
[[ -x "$PYTHON_BIN" ]] || die "Python not found: $PYTHON_BIN; run ./setup_env.sh"
[[ -c "$UGDS_DEVICE" ]] || die "$UGDS_DEVICE is not a character device"
[[ -r "$UGDS_DEVICE" && -w "$UGDS_DEVICE" ]] || die \
    "current user needs read/write permission on $UGDS_DEVICE"
[[ -f "$UGDS_LIB_DIR/libugds.so" ]] || die \
    "missing $UGDS_LIB_DIR/libugds.so; run ./setup_env.sh"
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"
command -v timeout >/dev/null 2>&1 || die "timeout is required"

for value in "$GPU_ID" "$MAX_MODEL_LEN" "$PROMPT_REPEATS" \
    "$VLLM_PORT" "$LMCACHE_PORT" "$LMCACHE_HTTP_PORT" "$STARTUP_TIMEOUT"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "expected a non-negative integer, got: $value"
done
[[ "$L1_SIZE_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die \
    "L1_SIZE_GB must be a positive number"
[[ ! "$L1_SIZE_GB" =~ ^0+([.]0+)?$ ]] || die \
    "L1_SIZE_GB must be greater than zero"
((MAX_MODEL_LEN > 0)) || die "MAX_MODEL_LEN must be greater than zero"
((PROMPT_REPEATS > 0)) || die "PROMPT_REPEATS must be greater than zero"
((STARTUP_TIMEOUT > 0)) || die "STARTUP_TIMEOUT must be greater than zero"
[[ "$VLLM_PORT" != "$LMCACHE_PORT" && \
    "$VLLM_PORT" != "$LMCACHE_HTTP_PORT" && \
    "$LMCACHE_PORT" != "$LMCACHE_HTTP_PORT" ]] || die \
    "VLLM_PORT, LMCACHE_PORT, and LMCACHE_HTTP_PORT must be distinct"

for port in "$VLLM_PORT" "$LMCACHE_PORT" "$LMCACHE_HTTP_PORT"; do
    ((port >= 1 && port <= 65535)) || die "invalid TCP port: $port"
    port_is_busy "$port" && die "TCP port $port is already in use"
done

mkdir -p "$ARTIFACT_DIR"
if [[ "$USE_LOCAL_VLLM_SOURCE" == "1" ]]; then
    export PYTHONPATH="$LMCACHE_REPO:$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
elif [[ "$USE_LOCAL_VLLM_SOURCE" == "0" ]]; then
    export PYTHONPATH="$LMCACHE_REPO${PYTHONPATH:+:$PYTHONPATH}"
else
    die "USE_LOCAL_VLLM_SOURCE must be 0 or 1"
fi
export LD_LIBRARY_PATH="$UGDS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export BENCH_LMCACHE_REPO="$LMCACHE_REPO"
export BENCH_VLLM_REPO="$VLLM_REPO"
export BENCH_USE_LOCAL_VLLM="$USE_LOCAL_VLLM_SOURCE"

printf '==> Preflight checks\n'
"$PYTHON_BIN" -c \
    'import ctypes; lib = ctypes.CDLL("libugds.so"); assert hasattr(lib, "uGDSGetDeviceCapacity")'
"$PYTHON_BIN" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
{
    printf 'LMCache revision: '
    git -C "$LMCACHE_REPO" describe --always --dirty
    printf 'vLLM revision: '
    git -C "$VLLM_REPO" describe --always --dirty
    printf 'uGDS revision: '
    git -C "$UGDS_REPO" describe --always --dirty
    "$PYTHON_BIN" - <<'PY'
import os
import pathlib

import lmcache
import vllm

lmcache_repo = pathlib.Path(os.environ["BENCH_LMCACHE_REPO"])
vllm_repo = pathlib.Path(os.environ["BENCH_VLLM_REPO"])
lmcache_path = pathlib.Path(lmcache.__file__).resolve()
vllm_path = pathlib.Path(vllm.__file__).resolve()
assert lmcache_path.is_relative_to(lmcache_repo), lmcache_path
if os.environ["BENCH_USE_LOCAL_VLLM"] == "1":
    assert vllm_path.is_relative_to(vllm_repo), vllm_path
print("lmcache:", lmcache_path)
print("vllm:", vllm_path)
PY
} | tee "$ARTIFACT_DIR/versions.txt"

if [[ -z "$LMCACHE_CONNECTOR_MODULE" ]]; then
    VLLM_VERSION="$($PYTHON_BIN -c \
        'import importlib.metadata as m; print(m.version("vllm"))')"
    case "$VLLM_VERSION" in
        0.20.1*)
            LMCACHE_CONNECTOR_MODULE="lmcache.integration.vllm.lmcache_mp_connector_0201"
            ;;
        0.18.*)
            LMCACHE_CONNECTOR_MODULE="lmcache.integration.vllm.lmcache_mp_connector_0180"
            ;;
        *)
            LMCACHE_CONNECTOR_MODULE="lmcache.integration.vllm.lmcache_mp_connector"
            ;;
    esac
fi
printf 'LMCache connector: %s\n' "$LMCACHE_CONNECTOR_MODULE" | \
    tee -a "$ARTIFACT_DIR/versions.txt"

printf '==> Starting LMCache with uGDS L1 on %s\n' "$UGDS_DEVICE"
setsid "$PYTHON_BIN" -m lmcache.cli.main server \
    --host 127.0.0.1 \
    --port "$LMCACHE_PORT" \
    --http-host 127.0.0.1 \
    --http-port "$LMCACHE_HTTP_PORT" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --chunk-size 256 \
    --gds-l1-backend ugds \
    --gds-l1-path "$UGDS_DEVICE" \
    >"$ARTIFACT_DIR/lmcache.log" 2>&1 &
LMCACHE_PID=$!
wait_for_url \
    "http://127.0.0.1:$LMCACHE_HTTP_PORT/healthcheck" \
    "$LMCACHE_PID" "LMCache" 120
grep -q 'GDSContext: uGDS raw-device slab opened' \
    "$ARTIFACT_DIR/lmcache.log" || die \
    "LMCache became healthy without logging uGDS raw-device initialization"
curl --fail --silent --show-error \
    "http://127.0.0.1:$LMCACHE_HTTP_PORT/status" \
    >"$ARTIFACT_DIR/lmcache-status.json"

KV_TRANSFER_CONFIG="$(printf \
    '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"%s\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":%s}}' \
    "$LMCACHE_CONNECTOR_MODULE" "$LMCACHE_PORT")"

printf '==> Starting vLLM model %s\n' "$MODEL"
setsid "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --served-model-name "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.80 \
    --no-enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    >"$ARTIFACT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
wait_for_url \
    "http://127.0.0.1:$VLLM_PORT/health" \
    "$VLLM_PID" "vLLM" "$STARTUP_TIMEOUT"

printf '==> Running cold/warm request benchmark\n'
"$PYTHON_BIN" "$BENCH_DIR/bench_client.py" \
    --base-url "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL" \
    --prompt-repeats "$PROMPT_REPEATS" \
    --output "$ARTIFACT_DIR/bench-result.json"

printf '\nArtifacts: %s\n' "$ARTIFACT_DIR"
printf 'LMCache uGDS evidence:\n'
grep -Ei 'ugds|gds l1' "$ARTIFACT_DIR/lmcache.log" | tail -n 10 || true
