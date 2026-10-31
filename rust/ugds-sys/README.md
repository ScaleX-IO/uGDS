# ugds-sys

Raw Rust declarations for [`include/ugds.h`](../../include/ugds.h) and a
runtime loader for `libugds.so`.

This crate contains no safe wrappers or policy. Higher-level crates own buffer
lifetimes, completion and error handling on top of these declarations.

## Usage

```rust
use ugds_sys::{libloading, uGDSOpError, UgdsApi};

fn main() -> Result<(), libloading::Error> {
    // SAFETY: loading runs the library's initialization routines.
    let api = unsafe { UgdsApi::load_default()? };

    // Each function is resolved independently; check optional APIs before use.
    if api.uGDSBatchIOSubmit.is_ok() {
        // SAFETY: follows the contract in ugds.h.
        let status = unsafe { api.uGDSDriverOpen() };
        assert_eq!(status.err, uGDSOpError::UGDS_SUCCESS);
    }
    Ok(())
}
```

`UgdsApi::new(path)` loads a library from an explicit path. Nothing is linked
at build time, so the crate builds without uGDS, CUDA or a GPU. The generated
`UgdsApi::uGDS*` methods panic if their symbol is missing; check the matching
`Result` field first when a function may be absent.

C enums are `#[repr(transparent)]` newtypes with associated constants, so
values outside the declared set remain representable. `ssize_t` maps to
`isize`; `off_t` and `timespec` come from `libc`.

## Regenerating

`src/bindings.rs` is generated. After changing `include/ugds.h`, run:

```sh
cargo install bindgen-cli --version 0.73.2 --locked
rust/ugds-sys/regen.sh
```

CI regenerates the bindings on Linux and fails if the checked-in file differs.

## Versioning

The crate version follows the C ABI of `ugds.h`: any header change that alters
declarations requires a crate version bump. Publication to crates.io is not
enabled yet.

## Maintenance

Maintained by [OrbitInfra](https://github.com/OrbitInfra) together with the
uGDS maintainers. Licensed under the repository's BSD 3-Clause license.
