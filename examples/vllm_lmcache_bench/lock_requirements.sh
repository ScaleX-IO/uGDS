#!/usr/bin/env bash
set -Eeuo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UGDS_REPO="${UGDS_REPO:-$(cd -- "$BENCH_DIR/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "$UGDS_REPO")}"
LMCACHE_REPO="${LMCACHE_REPO:-$WORKSPACE_DIR/LMCache}"
VLLM_REPO="${VLLM_REPO:-$WORKSPACE_DIR/vllm}"
USER_DATA_ROOT="${XDG_DATA_HOME:-${HOME:?HOME is required}/.local/share}"
USER_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}"
UV_TOOLS_DIR="${UV_TOOLS_DIR:-$USER_DATA_ROOT/ugds-bench/vllm_lmcache_bench/tools}"
UV_BIN="${UV_BIN:-}"
REQUIREMENTS_IN="${REQUIREMENTS_IN:-$BENCH_DIR/requirements.in}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$BENCH_DIR/requirements.txt}"
TEMP_REQUIREMENTS="$(mktemp --suffix=.txt)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ -z "$UV_BIN" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "$UV_TOOLS_DIR/uv" ]]; then
        UV_BIN="$UV_TOOLS_DIR/uv"
    elif [[ -x "$BENCH_DIR/.tools/uv" ]]; then
        UV_BIN="$BENCH_DIR/.tools/uv"
    else
        die "uv is not installed; run ./setup_env.sh first or set UV_BIN"
    fi
fi

[[ -x "$UV_BIN" ]] || command -v "$UV_BIN" >/dev/null 2>&1 || die \
    "UV_BIN is not executable: $UV_BIN"
[[ -f "$VLLM_REPO/vllm/_version.py" ]] || die "invalid vLLM repo: $VLLM_REPO"
[[ -f "$LMCACHE_REPO/lmcache/_version.py" ]] || die \
    "invalid LMCache repo: $LMCACHE_REPO"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$USER_CACHE_ROOT/ugds-bench/vllm_lmcache_bench/uv}"

"$UV_BIN" pip compile "$REQUIREMENTS_IN" \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 \
    --torch-backend cu129 \
    --output-file "$TEMP_REQUIREMENTS" \
    --custom-compile-command './lock_requirements.sh'

VLLM_VERSION="$(sed -n "s/^__version__ = version = '\([^']*\)'.*/\1/p" \
    "$VLLM_REPO/vllm/_version.py")"
LMCACHE_VERSION="$(sed -n "s/^__version__ = version = '\([^']*\)'.*/\1/p" \
    "$LMCACHE_REPO/lmcache/_version.py")"
VLLM_REVISION="$(git -C "$VLLM_REPO" rev-parse HEAD)"
LMCACHE_REVISION="$(git -C "$LMCACHE_REPO" rev-parse HEAD)"

[[ -n "$VLLM_VERSION" ]] || die "could not read the vLLM package version"
[[ -n "$LMCACHE_VERSION" ]] || die "could not read the LMCache package version"

{
    sed -n '1,2p' "$TEMP_REQUIREMENTS"
    printf '%s\n' \
        '#' \
        '# Local editable sources installed separately by setup_env.sh:' \
        "# vllm==$VLLM_VERSION" \
        "#   commit $VLLM_REVISION" \
        "# lmcache==$LMCACHE_VERSION" \
        "#   commit $LMCACHE_REVISION"
    sed -n '3,$p' "$TEMP_REQUIREMENTS"
} > "$REQUIREMENTS_FILE"

printf 'Locked dependencies and source revisions in %s\n' "$REQUIREMENTS_FILE"
