#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    uGDSBatchHandle_t batch = nullptr;

    // 1. SetUp: null batch pointer
    st = uGDSBatchIOSetUp(nullptr, fh, 4);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "SetUp nullptr batch");

    // 2. SetUp: null handle
    st = uGDSBatchIOSetUp(&batch, nullptr, 4);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "SetUp nullptr fh");

    // 3. SetUp: nr = 0
    st = uGDSBatchIOSetUp(&batch, fh, 0);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "SetUp nr=0");

    // 4. SetUp: valid — should succeed
    st = uGDSBatchIOSetUp(&batch, fh, 4);
    ASSERT_OK(st, "SetUp valid");

    // 5. Submit: null batch
    uGDSIOParams_t params;
    memset(&params, 0, sizeof(params));
    st = uGDSBatchIOSubmit(nullptr, 1, &params, 0);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "Submit nullptr batch");

    // 6. Submit: null iocb
    st = uGDSBatchIOSubmit(batch, 1, nullptr, 0);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "Submit nullptr iocb");

    // 7. Submit: nr = 0
    st = uGDSBatchIOSubmit(batch, 0, &params, 0);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "Submit nr=0");

    // 8. Submit: nr > capacity
    uGDSIOParams_t params5[5];
    memset(params5, 0, sizeof(params5));
    st = uGDSBatchIOSubmit(batch, 5, params5, 0);
    ASSERT_ERR(st, UGDS_BATCH_CAPACITY_EXCEEDED, "Submit nr>capacity");

    // 9. GetStatus: null batch
    unsigned nr_out = 0;
    uGDSIOEvents_t ev;
    st = uGDSBatchIOGetStatus(nullptr, 0, &nr_out, &ev, nullptr);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "GetStatus nullptr batch");

    // 10. GetStatus: null nr
    st = uGDSBatchIOGetStatus(batch, 0, nullptr, &ev, nullptr);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "GetStatus nullptr nr");

    // 11. GetStatus: null events
    st = uGDSBatchIOGetStatus(batch, 0, &nr_out, nullptr, nullptr);
    ASSERT_ERR(st, UGDS_INVALID_VALUE, "GetStatus nullptr events");

    // 12. Destroy: null should not crash
    uGDSBatchIODestroy(nullptr);

    // 13. Destroy: valid
    uGDSBatchIODestroy(batch);

    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
