#include "test_utils.h"
#include <errno.h>

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t alloc_size = 65536;
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, alloc_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, alloc_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    ssize_t ret;

    // 1. Non-block-aligned file_offset
    ret = uGDSRead(fh, d_buf, 4096, 100, 0);
    if (ret != -EINVAL)
        TEST_FAIL("unaligned offset: expected %d, got %zd", -EINVAL, ret);

    // 2. Non-block-aligned size
    ret = uGDSRead(fh, d_buf, 100, 0, 0);
    if (ret != -EINVAL)
        TEST_FAIL("unaligned size: expected %d, got %zd", -EINVAL, ret);

    // 3. Zero size
    ret = uGDSRead(fh, d_buf, 0, 0, 0);
    if (ret != -EINVAL)
        TEST_FAIL("zero size: expected %d, got %zd", -EINVAL, ret);

    // 4. Null buffer
    ret = uGDSRead(fh, nullptr, 4096, 0, 0);
    if (ret != -EINVAL)
        TEST_FAIL("null buffer: expected %d, got %zd", -EINVAL, ret);

    // 5. Negative offset (cast to off_t)
    ret = uGDSRead(fh, d_buf, 4096, (off_t)-1, 0);
    if (ret != -EINVAL)
        TEST_FAIL("negative offset: expected %d, got %zd", -EINVAL, ret);

    // 6. On-the-fly buffer must be 64KB-aligned (kernel rounds down otherwise)
    const uintptr_t kGpuPageSize = 1UL << 16;
    void* d_unreg = nullptr;
    cudaMalloc(&d_unreg, 131072);
    if (!d_unreg) TEST_FAIL("cudaMalloc (unregistered) failed");
    // Pick a buf offset that guarantees base+offset is NOT 64KB-aligned,
    // regardless of what alignment cudaMalloc happened to return.
    uintptr_t base = (uintptr_t)d_unreg;
    off_t mis_off = ((base & (kGpuPageSize - 1)) == 0) ? 4096 : 0;
    ret = uGDSRead(fh, d_unreg, 4096, 0, mis_off);
    if (ret != -EINVAL)
        TEST_FAIL("unaligned on-the-fly buffer: expected %d, got %zd", -EINVAL, ret);
    cudaFree(d_unreg);

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
