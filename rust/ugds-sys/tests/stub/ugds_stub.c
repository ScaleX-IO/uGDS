/* Minimal library for the loader test. It exports only some uGDS symbols,
 * so the test can check per-function resolution and values crossing the ABI. */
#include "ugds.h"

uGDSError_t uGDSDriverOpen(void)
{
    uGDSError_t status = {UGDS_SUCCESS, 0};
    return status;
}

uGDSError_t uGDSDriverClose(void)
{
    uGDSError_t status = {UGDS_BUSY, 7};
    return status;
}

uGDSError_t uGDSGetDeviceCapacity(uGDSHandle_t fh, uint64_t* capacity_bytes)
{
    (void)fh;
    *capacity_bytes = 0x123456789abcULL;
    uGDSError_t status = {UGDS_SUCCESS, 0};
    return status;
}

uGDSError_t uGDSBatchIOGetStatus(uGDSBatchHandle_t batch, unsigned min_nr,
                                  unsigned* nr, uGDSIOEvents_t* events,
                                  struct timespec* timeout)
{
    (void)batch;
    (void)min_nr;
    (void)timeout;
    events[0].cookie = (void*)0x42;
    events[0].status = UGDS_BATCH_COMPLETE;
    events[0].ret = 4096;
    *nr = 1;
    uGDSError_t status = {UGDS_SUCCESS, 0};
    return status;
}
