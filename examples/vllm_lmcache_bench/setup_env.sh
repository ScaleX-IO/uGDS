#!/usr/bin/env bash
set -Eeuo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UGDS_REPO="${UGDS_REPO:-$(cd -- "$BENCH_DIR/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$UGDS_REPO")}"
LMCACHE_REPO="${LMCACHE_REPO:-$WORKSPACE_DIR/LMCache}"
VLLM_REPO="${VLLM_REPO:-$WORKSPACE_DIR/vllm}"
VENV_DIR="${VENV_DIR:-$BENCH_DIR/.venv}"
UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
EXISTING_PYTHON="${EXISTING_PYTHON:-}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -f "$UGDS_REPO/CMakeLists.txt" ]] || die "invalid uGDS repo: $UGDS_REPO"
[[ -f "$LMCACHE_REPO/pyproject.toml" ]] || die \
    "LMCache repo not found: $LMCACHE_REPO (set LMCACHE_REPO)"
[[ -f "$VLLM_REPO/pyproject.toml" ]] || die \
    "vLLM repo not found: $VLLM_REPO (set VLLM_REPO)"

command -v "$UV_BIN" >/dev/null 2>&1 || die \
    "uv is required; install it from https://docs.astral.sh/uv/"
command -v cmake >/dev/null 2>&1 || die "cmake is required to build uGDS"

if [[ -n "$EXISTING_PYTHON" ]]; then
    [[ -x "$EXISTING_PYTHON" ]] || die \
        "EXISTING_PYTHON is not executable: $EXISTING_PYTHON"
    PYTHON_BIN="$EXISTING_PYTHON"
    printf '==> Reusing existing Python environment: %s\n' "$PYTHON_BIN"
    "$PYTHON_BIN" -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'

    printf '==> Rebuilding only local LMCache (dependencies unchanged)\n'
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH" \
    "$UV_BIN" pip install --python "$PYTHON_BIN" --no-deps \
        -e "$LMCACHE_REPO" --no-build-isolation
else
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        printf '==> Reusing Python environment at %s\n' "$VENV_DIR"
    else
        printf '==> Creating Python %s environment at %s\n' \
            "$PYTHON_VERSION" "$VENV_DIR"
        "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
    fi
    PYTHON_BIN="$VENV_DIR/bin/python"

    printf '==> Installing PyTorch\n'
    "$UV_BIN" pip install --python "$PYTHON_BIN" torch --torch-backend=auto

    printf '==> Installing native-extension build requirements\n'
    "$UV_BIN" pip install --python "$PYTHON_BIN" \
        ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' \
        'setuptools-scm>=8' wheel

    printf '==> Installing local vLLM source (precompiled extension)\n'
    VLLM_USE_PRECOMPILED=1 "$UV_BIN" pip install \
        --python "$PYTHON_BIN" -e "$VLLM_REPO" --torch-backend=auto

    printf '==> Installing local LMCache source and CUDA extension\n'
    "$UV_BIN" pip install --python "$PYTHON_BIN" \
        -e "$LMCACHE_REPO" --no-build-isolation
fi

printf '==> Building libugds.so\n'
cmake -S "$UGDS_REPO" -B "$UGDS_REPO/build" \
    -DUGDS_BACKEND_CUDA=ON -DUGDS_BACKEND_HIP=OFF
cmake --build "$UGDS_REPO/build" --target ugds -j "$(nproc)"

if [[ "${BUILD_UGDS_DRIVER:-0}" == "1" ]]; then
    printf '==> Building ugds_drv.ko\n'
    make -C "$UGDS_REPO/drv" BUILD_CUDA=1 HAVE_CUDA_DMABUF=1
fi

if [[ -n "$EXISTING_PYTHON" ]]; then
    VALIDATION_PYTHONPATH="$LMCACHE_REPO"
else
    VALIDATION_PYTHONPATH="$LMCACHE_REPO:$VLLM_REPO"
fi
PYTHONPATH="$VALIDATION_PYTHONPATH" "$PYTHON_BIN" -c \
    'import lmcache, torch, vllm; print("torch", torch.__version__); print("vllm", vllm.__version__); print("lmcache", lmcache.__file__)'

printf '%s\n' "$PYTHON_BIN" > "$BENCH_DIR/.python-bin"

printf '\nEnvironment is ready. Next, bind a dedicated NVMe device and run:\n'
printf '  UGDS_PCI_SLOT=<slot> I_UNDERSTAND_UGDS_ERASES_DEVICE=1 ./quickstart.sh\n'
