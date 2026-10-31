#!/usr/bin/env bash
# Regenerate src/bindings.rs from include/ugds.h.
# CI runs this with the pinned bindgen and fails if the output changes.
# prettyplease keeps the output independent of the installed rustfmt.
set -euo pipefail

BINDGEN_VERSION="0.73.2"
crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${crate_dir}/../.." && pwd)"

if ! bindgen --version | grep -qx "bindgen ${BINDGEN_VERSION}"; then
    echo "error: bindgen ${BINDGEN_VERSION} is required" >&2
    echo "install it with: cargo install bindgen-cli --version ${BINDGEN_VERSION} --locked" >&2
    exit 1
fi

bindgen "${repo_dir}/include/ugds.h" \
    --rust-target 1.85 \
    --rust-edition 2021 \
    --formatter prettyplease \
    --dynamic-loading UgdsApi \
    --allowlist-function 'uGDS.*' \
    --allowlist-type 'uGDS.*' \
    --allowlist-var 'UGDS_.*|NVM_MAP_.*' \
    --newtype-enum 'uGDS.*' \
    --blocklist-type 'off_t|timespec|time_t|__.*' \
    --raw-line 'use libc::{off_t, timespec};' \
    --output "${crate_dir}/src/bindings.rs" \
    -- -I"${repo_dir}/include"
