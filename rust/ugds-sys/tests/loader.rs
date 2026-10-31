use std::path::PathBuf;
use std::process::Command;
use std::ptr::null_mut;
use std::sync::OnceLock;

use ugds_sys::{uGDSBatchStatus, uGDSIOEvents, uGDSOpError, UgdsApi};

/// Builds tests/stub/ugds_stub.c into a shared library once per test binary.
fn stub_library() -> &'static PathBuf {
    static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("libugds_stub.so");
        let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_owned());
        let status = Command::new(&compiler)
            .args(["-shared", "-fPIC", "-I"])
            .arg(crate_dir.join("../../include"))
            .arg(crate_dir.join("tests/stub/ugds_stub.c"))
            .arg("-o")
            .arg(&output)
            .status()
            .unwrap_or_else(|error| panic!("cannot run C compiler {compiler:?}: {error}"));
        assert!(
            status.success(),
            "compiling the loader stub failed: {status}"
        );
        output
    })
}

fn load_stub() -> UgdsApi {
    // SAFETY: the stub has no initialization routines.
    unsafe { UgdsApi::new(stub_library()) }.expect("loading the stub library")
}

#[test]
fn each_function_reports_its_own_resolution() {
    let api = load_stub();

    assert!(api.uGDSDriverOpen.is_ok());
    assert!(api.uGDSGetDeviceCapacity.is_ok());
    assert!(api.uGDSBatchIOGetStatus.is_ok());
    assert!(api.uGDSBatchIOSubmit.is_err());
    assert!(api.uGDSReadAsync.is_err());
}

#[test]
fn values_cross_the_c_abi() {
    let api = load_stub();

    // SAFETY: each stub function only writes through the provided pointers.
    unsafe {
        let opened = api.uGDSDriverOpen();
        assert_eq!((opened.err, opened.cu_err), (uGDSOpError::UGDS_SUCCESS, 0));
        let closed = api.uGDSDriverClose();
        assert_eq!((closed.err, closed.cu_err), (uGDSOpError::UGDS_BUSY, 7));

        let mut capacity = 0;
        api.uGDSGetDeviceCapacity(null_mut(), &mut capacity);
        assert_eq!(capacity, 0x1234_5678_9abc);

        let mut events = [uGDSIOEvents {
            cookie: null_mut(),
            status: uGDSBatchStatus(0),
            ret: 0,
        }; 2];
        let mut completed = 0;
        api.uGDSBatchIOGetStatus(
            null_mut(),
            1,
            &mut completed,
            events.as_mut_ptr(),
            null_mut(),
        );
        assert_eq!(completed, 1);
        assert_eq!(events[0].cookie as usize, 0x42);
        assert_eq!(events[0].status, uGDSBatchStatus::UGDS_BATCH_COMPLETE);
        assert_eq!(events[0].ret, 4096);
    }
}
