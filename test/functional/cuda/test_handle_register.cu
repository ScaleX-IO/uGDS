#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    // 1. Register a valid handle
    int fd = open(g_dev_path, O_RDWR);
    if (fd < 0) TEST_FAIL("open(%s) failed", g_dev_path);

    uGDSDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;

    uGDSHandle_t fh = nullptr;
    st = uGDSHandleRegister(&fh, &descr);
    ASSERT_OK(st, "HandleRegister");

    // 2. Deregister
    uGDSHandleDeregister(fh);

    // 3. Register again on the same fd (reuse after deregister)
    fh = nullptr;
    st = uGDSHandleRegister(&fh, &descr);
    ASSERT_OK(st, "HandleRegister after deregister");
    uGDSHandleDeregister(fh);
    close(fd);

    // 4. nullptr fh pointer -> INVALID_VALUE
    st = uGDSHandleRegister(nullptr, &descr);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "nullptr fh");

    // 5. nullptr descr -> INVALID_VALUE
    uGDSHandle_t fh2 = nullptr;
    st = uGDSHandleRegister(&fh2, nullptr);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "nullptr descr");

    // 6. Invalid handle type (WIN32) -> INVALID_FILE_TYPE
    fd = open(g_dev_path, O_RDWR);
    if (fd < 0) TEST_FAIL("open(%s) failed for win32 test", g_dev_path);
    uGDSDescr_t bad_descr;
    memset(&bad_descr, 0, sizeof(bad_descr));
    bad_descr.type = UGDS_HANDLE_TYPE_OPAQUE_WIN32;
    bad_descr.handle.fd = fd;
    fh2 = nullptr;
    st = uGDSHandleRegister(&fh2, &bad_descr);
    ASSERT_ERR(st, UGDS_INVALID_FILE_TYPE, "WIN32 handle type");
    close(fd);

    // 7. Deregister nullptr should not crash
    uGDSHandleDeregister(nullptr);

    uGDSDriverClose();
    TEST_PASS();
}
