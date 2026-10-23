#include "test_utils.h"

#include <pthread.h>

struct ReadArgs {
    uGDSHandle_t handle;
    void* buffer;
    off_t buffer_offset; /* Destination offset in the GPU buffer */
    ssize_t result;
};

/* This is a thread entry function that conforms to the POSIX `pthread` interface requirements.
 * Each thread calls `uGDSRead()` once:
   - Read length: 4096 bytes (4 KiB).
   - Device offset: 0.
   - Buffer offset: Specified by the corresponding `ReadArgs`.
 * After the function completes, the return value is stored in its own `result` field. */
static void* read_one(void* arg)
{
    ReadArgs* args = static_cast<ReadArgs*>(arg);
    args->result = uGDSRead(args->handle, args->buffer, 4096, 0,
                            args->buffer_offset);
    return nullptr;
}

int main(int argc, char** argv)
{
    if (!parse_args(argc, argv)) return 1;
    const char* second_path = getenv("UGDS_MULTI_SSD_DEVICE_2");
    /* access() checks whether the path has read and write access permissions. */
    if ((second_path == nullptr || second_path[0] == 0) &&
        access("/dev/ugds_drv1", R_OK | W_OK) == 0)
        second_path = "/dev/ugds_drv1";
    if (second_path == nullptr || second_path[0] == 0) {
        /* The second-device path has not been tested on this host yet. */
        fprintf(stderr, "SKIP: set UGDS_MULTI_SSD_DEVICE_2 to a second uGDS device\n");
        return 77;
    }
    /* Strictly speaking, different paths do not necessarily represent different physical devices; 
     * two different paths may point to the same underlying device. However, 
     * this test at least rules out the most straightforward case of duplicate paths. */
    if (strcmp(second_path, g_dev_path) == 0) {
        fprintf(stderr, "SKIP: multi-SSD test requires two different device paths\n");
        return 77;
    }

    cudaSetDevice(g_gpu_id);
    ASSERT_OK(uGDSDriverOpen(), "DriverOpen");
    uGDSHandle_t first = open_handle();
    uGDSHandle_t second = open_handle_at(second_path);
    if (!first || !second) {
        close_handle(first);
        close_handle(second);
        uGDSDriverClose();
        fprintf(stderr, "SKIP: two independently registered uGDS devices are required\n");
        return 77;
    }

    /* Test Storage */
    uGDSStorage_t storage = nullptr;
    ASSERT_ERR(uGDSStorageCreate(nullptr), UGDS_INVALID_VALUE, "StorageCreate null");
    ASSERT_OK(uGDSStorageCreate(&storage), "StorageCreate");

    uint32_t count = 1;
    ASSERT_OK(uGDSStorageGetDeviceCount(storage, &count), "empty storage count");
    if (count != 0) TEST_FAIL("empty storage count: expected 0, got %u", count);
    uint64_t total_capacity = 1;
    ASSERT_OK(uGDSStorageGetCapacity(storage, &total_capacity), "empty storage capacity");
    if (total_capacity != 0) TEST_FAIL("empty storage capacity: expected 0");

    ASSERT_ERR(uGDSStorageAddHandle(storage, nullptr), UGDS_INVALID_VALUE,
               "StorageAddHandle null");
    ASSERT_OK(uGDSStorageAddHandle(storage, first), "StorageAddHandle first");
    ASSERT_ERR(uGDSStorageAddHandle(storage, first), UGDS_INVALID_VALUE,
               "StorageAddHandle duplicate");
    ASSERT_OK(uGDSStorageAddHandle(storage, second), "StorageAddHandle second");
    ASSERT_OK(uGDSStorageGetDeviceCount(storage, &count), "storage count");
    if (count != 2) TEST_FAIL("storage count: expected 2, got %u", count);

    uint64_t first_capacity = 0, second_capacity = 0;
    ASSERT_OK(uGDSGetDeviceCapacity(first, &first_capacity), "first handle capacity");
    ASSERT_OK(uGDSGetDeviceCapacity(second, &second_capacity), "second handle capacity");
    ASSERT_OK(uGDSStorageGetDeviceCapacity(storage, 0, &total_capacity),
              "storage first capacity");
    if (total_capacity != first_capacity) TEST_FAIL("first storage capacity mismatch");
    ASSERT_OK(uGDSStorageGetDeviceCapacity(storage, 1, &total_capacity),
              "storage second capacity");
    if (total_capacity != second_capacity) TEST_FAIL("second storage capacity mismatch");
    ASSERT_OK(uGDSStorageGetCapacity(storage, &total_capacity), "storage total capacity");
    if (total_capacity != first_capacity + second_capacity)
        TEST_FAIL("storage total capacity mismatch");

    uGDSHandle_t queried = nullptr;
    ASSERT_OK(uGDSStorageGetHandle(storage, 0, &queried), "storage first handle");
    if (queried != first) TEST_FAIL("storage first handle mismatch");
    ASSERT_OK(uGDSStorageGetHandle(storage, 1, &queried), "storage second handle");
    if (queried != second) TEST_FAIL("storage second handle mismatch");
    ASSERT_ERR(uGDSStorageGetHandle(storage, 2, &queried), UGDS_INVALID_VALUE,
               "storage out-of-range handle");

    /* Two threads read from different devices simultaneously. */
    void* d_buf = nullptr; /* Device buffer */
    cudaMalloc(&d_buf, 65536); /* 64 KiB */
    if (!d_buf) TEST_FAIL("cudaMalloc failed");
    ASSERT_OK(uGDSBufRegister(d_buf, 65536, TEST_BUF_FLAGS), "BufRegister");
    /* The two threads write to separate 4 KiB regions, 
     * avoiding overlap in the destination buffer. */
    ReadArgs first_read = {first, d_buf, 0, 0};
    ReadArgs second_read = {second, d_buf, 4096, 0};
    pthread_t first_thread, second_thread;
    int first_rc = pthread_create(
        &first_thread, nullptr, read_one, &first_read
    );

    int second_rc = -1;

    if (first_rc == 0) {
        second_rc = pthread_create(
            &second_thread, nullptr, read_one, &second_read
        );
    }
    if (first_rc != 0 || second_rc != 0) {
        if (first_rc == 0) pthread_join(first_thread, nullptr);
        TEST_FAIL("pthread_create failed");
    }
    /* The main thread waits for both read threads to complete. */
    pthread_join(first_thread, nullptr);
    pthread_join(second_thread, nullptr);
    if (first_read.result != 4096 || second_read.result != 4096)
        TEST_FAIL("independent-device reads failed: %zd, %zd",
                  first_read.result, second_read.result);
    /* Lifecycle test: A Handle cannot be unregistered while it is held by a Storage. */
    ASSERT_ERR(uGDSHandleDeregisterEx(first, 0), UGDS_BUSY,
               "storage retains first handle");

    ASSERT_OK(uGDSBufDeregister(d_buf), "BufDeregister");
    cudaFree(d_buf);
    uGDSStorageDestroy(storage);
    close_handle(first);
    close_handle(second);
    uGDSDriverClose();
    TEST_PASS();
}
