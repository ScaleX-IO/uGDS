#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    // Open a handle first so default_ctrl is initialized
    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t buf_size = 65536;
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, buf_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    // 1. Register -> OK
    st = uGDSBufRegister(d_buf, buf_size, 0);
    ASSERT_OK(st, "BufRegister");

    // 2. Double register -> MEMORY_ALREADY_REGISTERED
    st = uGDSBufRegister(d_buf, buf_size, 0);
    ASSERT_ERR(st, UGDS_MEMORY_ALREADY_REGISTERED, "double BufRegister");

    // 3. Deregister -> OK
    st = uGDSBufDeregister(d_buf);
    ASSERT_OK(st, "BufDeregister");

    // 4. Deregister again -> MEMORY_NOT_REGISTERED
    st = uGDSBufDeregister(d_buf);
    ASSERT_ERR(st, UGDS_MEMORY_NOT_REGISTERED, "double BufDeregister");

    // 5. Re-register after deregister -> OK
    st = uGDSBufRegister(d_buf, buf_size, 0);
    ASSERT_OK(st, "re-register after deregister");
    uGDSBufDeregister(d_buf);

    // 6. Null ptr -> INVALID_VALUE
    st = uGDSBufRegister(nullptr, buf_size, 0);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "null ptr BufRegister");

    // 7. Zero length -> INVALID_VALUE
    st = uGDSBufRegister(d_buf, 0, 0);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "zero length BufRegister");

    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
