#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st;

    // 1. First open should succeed
    st = uGDSDriverOpen();
    ASSERT_OK(st, "first DriverOpen");

    // 2. Second open should be idempotent (succeed or ALREADY_OPEN)
    st = uGDSDriverOpen();
    if (st.err != UGDS_SUCCESS && st.err != UGDS_DRIVER_ALREADY_OPEN)
        TEST_FAIL("second DriverOpen: unexpected %s", uGDS_status_error(st.err));

    // 3. Close should succeed
    st = uGDSDriverClose();
    ASSERT_OK(st, "first DriverClose");

    // 4. Close again should return DRIVER_NOT_INITIALIZED
    st = uGDSDriverClose();
    ASSERT_ERR(st, UGDS_DRIVER_NOT_INITIALIZED, "second DriverClose");

    // 5. Reopen after close should succeed
    st = uGDSDriverOpen();
    ASSERT_OK(st, "reopen DriverOpen");

    uGDSDriverClose();
    TEST_PASS();
}
