#!/usr/bin/env bash
set -Eeuo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UGDS_REPO="${UGDS_REPO:-$(cd -- "$BENCH_DIR/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$UGDS_REPO")}"
LMCACHE_REPO="${LMCACHE_REPO:-$WORKSPACE_DIR/LMCache}"
VLLM_REPO="${VLLM_REPO:-$WORKSPACE_DIR/vllm}"
USER_DATA_ROOT="${XDG_DATA_HOME:-${HOME:?HOME is required}/.local/share}"
USER_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
VENV_DIR="${VENV_DIR:-$USER_DATA_ROOT/ugds-bench/vllm_lmcache_bench/.venv}"
UV_TOOLS_DIR="${UV_TOOLS_DIR:-$USER_DATA_ROOT/ugds-bench/vllm_lmcache_bench/tools}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$BENCH_DIR/requirements.txt}"
UV_BIN="${UV_BIN:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$USER_CACHE_ROOT/ugds-bench/vllm_lmcache_bench/uv}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

ensure_uv() {
    if [[ -n "$UV_BIN" ]]; then
        [[ -x "$UV_BIN" ]] || command -v "$UV_BIN" >/dev/null 2>&1 || die \
            "UV_BIN is not executable: $UV_BIN"
        printf '%s\n' "$UV_BIN"
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    if [[ -x "$UV_TOOLS_DIR/uv" ]]; then
        printf '%s\n' "$UV_TOOLS_DIR/uv"
        return
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
    printf '==> Installing uv for the current user under %s\n' \
        "$UV_TOOLS_DIR" >&2
    local installer
    installer="$(mktemp)"
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    UV_UNMANAGED_INSTALL="$UV_TOOLS_DIR" sh "$installer"
    rm -f "$installer"
    printf '%s\n' "$UV_TOOLS_DIR/uv"
}

[[ -f "$UGDS_REPO/CMakeLists.txt" ]] || die "invalid uGDS repo: $UGDS_REPO"
[[ -f "$LMCACHE_REPO/pyproject.toml" ]] || die \
    "LMCache repo not found: $LMCACHE_REPO (set LMCACHE_REPO)"
[[ -f "$VLLM_REPO/pyproject.toml" ]] || die \
    "vLLM repo not found: $VLLM_REPO (set VLLM_REPO)"
[[ -f "$REQUIREMENTS_FILE" ]] || die \
    "requirements lock not found: $REQUIREMENTS_FILE"
[[ -z "${EXISTING_PYTHON:-}" ]] || die \
    "EXISTING_PYTHON is no longer supported; set VENV_DIR for a user-owned environment"

UV_BIN="$(ensure_uv)"

if [[ -x "$VENV_DIR/bin/python" ]]; then
    printf '==> Reusing user environment at %s\n' "$VENV_DIR"
else
    printf '==> Creating user Python %s environment at %s\n' \
        "$PYTHON_VERSION" "$VENV_DIR"
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
PYTHON_BIN="$VENV_DIR/bin/python"

printf '==> Syncing pinned CUDA dependencies from %s\n' "$REQUIREMENTS_FILE"
"$UV_BIN" pip sync --python "$PYTHON_BIN" "$REQUIREMENTS_FILE" \
    --torch-backend="$TORCH_BACKEND"

printf '==> Installing local vLLM source (precompiled extension)\n'
VLLM_USE_PRECOMPILED=1 "$UV_BIN" pip install \
    --python "$PYTHON_BIN" --no-deps --no-build-isolation -e "$VLLM_REPO"

printf '==> Installing local LMCache source and CUDA extension\n'
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
LMCACHE_CUDA_MAJOR=12 \
PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$VENV_DIR/bin:$PATH" \
"$UV_BIN" pip install --python "$PYTHON_BIN" --no-deps \
    --no-build-isolation -e "$LMCACHE_REPO"

"$UV_BIN" pip check --python "$PYTHON_BIN"

printf '==> Building libugds.so\n'
"$VENV_DIR/bin/cmake" -S "$UGDS_REPO" -B "$UGDS_REPO/build" \
    -DUGDS_BACKEND_CUDA=ON -DUGDS_BACKEND_HIP=OFF
"$VENV_DIR/bin/cmake" --build "$UGDS_REPO/build" \
    --target ugds -j "$(nproc)"

if [[ "${BUILD_UGDS_DRIVER:-0}" == "1" ]]; then
    printf '==> Building ugds_drv.ko\n'
    make -C "$UGDS_REPO/drv" BUILD_CUDA=1 HAVE_CUDA_DMABUF=1
fi

PYTHONPATH="$LMCACHE_REPO:$VLLM_REPO" "$PYTHON_BIN" -c \
    'import lmcache, torch, vllm; print("torch", torch.__version__); print("vllm", vllm.__version__); print("lmcache", lmcache.__file__)'

printf '\nEnvironment is ready at %s. Next, bind a dedicated NVMe device and run:\n' \
    "$VENV_DIR"
printf '  UGDS_PCI_SLOT=<slot> I_UNDERSTAND_UGDS_ERASES_DEVICE=1 ./quickstart.sh\n'
