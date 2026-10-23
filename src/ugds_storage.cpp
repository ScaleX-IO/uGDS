#include "ugds_internal.h"

#include <limits>
#include <new>
/* Lookup a storage state by its handle */
static std::shared_ptr<StorageState> storage_lookup(uGDSStorage_t storage)
{
    if (storage == nullptr)
        return nullptr;
    std::lock_guard<std::mutex> guard(g_driver.lock);
    auto it = g_driver.storage_registry.find(
        static_cast<StorageState*>(storage));
    return it == g_driver.storage_registry.end() ? nullptr : it->second;
}

/* Create a new Storage object 
 * and write its handle to the output parameter provided by the caller. */
extern "C" uGDSError_t uGDSStorageCreate(uGDSStorage_t* storage)
{
    if (storage == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    if (!g_driver.initialized.load(std::memory_order_acquire))
        return make_error(UGDS_DRIVER_NOT_INITIALIZED);

    auto state = std::make_shared<StorageState>();
    StorageState* raw = state.get();
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        /* There is a time gap between the first initial check and the actual registration, 
         * during which the driver may be closed. */
        if (!g_driver.initialized.load(std::memory_order_acquire))
            return make_error(UGDS_DRIVER_NOT_INITIALIZED);
        try {
            g_driver.storage_registry.emplace(raw, state);
        } catch (const std::bad_alloc&) {
            return make_error(UGDS_OUT_OF_MEMORY);
        }
    }
    *storage = static_cast<uGDSStorage_t>(raw);
    return UGDS_OK;
}

/* Add a device to a Storage object. The parameters are:
 * storage: The target Storage object.
 * fh: A previously registered device handle.
 * Upon successful execution, 
 * one entry is added to the Storage object's device list, 
 * and its total capacity increases by the capacity of the added device. */
extern "C" uGDSError_t uGDSStorageAddHandle(uGDSStorage_t storage,
                                             uGDSHandle_t fh)
{
    /* Trade-off: Storage deduplicates entries by handle identity rather than
     * physical device identity. Different handles referring to the same SSD
     * may therefore be added as separate entries, potentially double-counting
     * capacity. Physical device deduplication is deferred to avoid introducing
     * additional device identity and handle lifetime management complexity. */
    if (storage == nullptr || fh == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    std::shared_ptr<StorageState> storage_sp = storage_lookup(storage);
    if (!storage_sp)
        return make_error(UGDS_INVALID_VALUE);

    std::shared_ptr<HandleState> handle_sp;
    HandleState* handle = handle_lookup(fh, &handle_sp);
    if (!handle)
        return make_error(UGDS_HANDLE_NOT_REGISTERED);

    /* HandleState's ns_info is fully initialized before it is registered
     * in handle_registry and is never modified after registration. The
     * in-flight reference acquired by handle_lookup() also prevents the
     * unregister path from destroying HandleState before this calculation
     * completes. Therefore, this capacity calculation does not need to
     * hold storage_sp->lock, and there is no risk of concurrent modification
     * of ns_info. */
    const uint64_t blocks = static_cast<uint64_t>(handle->ns_info.capacity);
    const uint64_t block_size =
        static_cast<uint64_t>(handle->ns_info.lba_data_size);
    if (blocks == 0 || block_size == 0 ||
        blocks > std::numeric_limits<uint64_t>::max() / block_size) {
        handle_release(handle);
        return make_error(UGDS_INTERNAL_ERROR);
    }
    const uint64_t capacity = blocks * block_size;

    {
        /* All error paths call `handle_release(handle)`, 
         * while the success path retains the reference for `Storage`.
         * In `uGDSStorageDestroy()`, the retained references are released one by one during destruction. */
        std::lock_guard<std::mutex> guard(storage_sp->lock);
        if (storage_sp->closing) {
            handle_release(handle);
            return make_error(UGDS_INVALID_VALUE);
        }
        for (const auto& device : storage_sp->devices) {
            if (device.handle == fh) {
                handle_release(handle);
                return make_error(UGDS_INVALID_VALUE);
            }
        }
        if (capacity > std::numeric_limits<uint64_t>::max() -
                           storage_sp->capacity_bytes) {
            handle_release(handle);
            return make_error(UGDS_INTERNAL_ERROR);
        }
        try {
            storage_sp->devices.push_back({fh, std::move(handle_sp), capacity});
        } catch (const std::bad_alloc&) {
            handle_release(handle);
            return make_error(UGDS_OUT_OF_MEMORY);
        }
        storage_sp->capacity_bytes += capacity;
    }
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStorageGetDeviceCount(uGDSStorage_t storage,
                                                   uint32_t* device_count)
{
    if (device_count == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    std::shared_ptr<StorageState> storage_sp = storage_lookup(storage);
    if (!storage_sp)
        return make_error(UGDS_INVALID_VALUE);
    std::lock_guard<std::mutex> guard(storage_sp->lock);
    if (storage_sp->closing)
        return make_error(UGDS_INVALID_VALUE);
    if (storage_sp->devices.size() > std::numeric_limits<uint32_t>::max())
        return make_error(UGDS_INTERNAL_ERROR);
    *device_count = static_cast<uint32_t>(storage_sp->devices.size());
    return UGDS_OK;
}

/* The following four functions follow a very similar pattern:
 * 1. Check whether the output parameter is null.
 * 2. Retrieve the internal object using `storage_lookup()`.
 * 3. Lock the Storage object.
 * 4. Check whether it is being closed.
 * 5. Return the corresponding information. */
extern "C" uGDSError_t uGDSStorageGetCapacity(uGDSStorage_t storage,
                                                uint64_t* capacity_bytes)
{
    if (capacity_bytes == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    std::shared_ptr<StorageState> storage_sp = storage_lookup(storage);
    if (!storage_sp)
        return make_error(UGDS_INVALID_VALUE);
    std::lock_guard<std::mutex> guard(storage_sp->lock);
    if (storage_sp->closing)
        return make_error(UGDS_INVALID_VALUE);
    *capacity_bytes = storage_sp->capacity_bytes;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStorageGetDeviceCapacity(
    uGDSStorage_t storage, uint32_t device_index, uint64_t* capacity_bytes)
{
    if (capacity_bytes == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    std::shared_ptr<StorageState> storage_sp = storage_lookup(storage);
    if (!storage_sp)
        return make_error(UGDS_INVALID_VALUE);
    std::lock_guard<std::mutex> guard(storage_sp->lock);
    if (storage_sp->closing || device_index >= storage_sp->devices.size())
        return make_error(UGDS_INVALID_VALUE);
    *capacity_bytes = storage_sp->devices[device_index].capacity_bytes;
    return UGDS_OK;
}

extern "C" uGDSError_t uGDSStorageGetHandle(uGDSStorage_t storage,
                                              uint32_t device_index,
                                              uGDSHandle_t* fh)
{
    if (fh == nullptr)
        return make_error(UGDS_INVALID_VALUE);
    std::shared_ptr<StorageState> storage_sp = storage_lookup(storage);
    if (!storage_sp)
        return make_error(UGDS_INVALID_VALUE);
    std::lock_guard<std::mutex> guard(storage_sp->lock);
    if (storage_sp->closing || device_index >= storage_sp->devices.size())
        return make_error(UGDS_INVALID_VALUE);
    *fh = storage_sp->devices[device_index].handle;
    return UGDS_OK;
}

/* Destroy Storage:
 * 1. Remove the entry from `storage_registry`.
 * 2. The local `storage_sp` continues to hold a reference to `StorageState`.
 * Prevent new calls from finding the Storage object being destroyed through the registry,
 * while allowing already-started concurrent calls to complete normally.
 * 3. Clean up the internal resources of the Storage object.
 * 4. The local shared pointer is released when the function returns. */
extern "C" void uGDSStorageDestroy(uGDSStorage_t storage)
{
    if (storage == nullptr)
        return;
    std::shared_ptr<StorageState> storage_sp;
    {
        std::lock_guard<std::mutex> guard(g_driver.lock);
        auto it = g_driver.storage_registry.find(
            static_cast<StorageState*>(storage));
        if (it == g_driver.storage_registry.end())
            return;
        storage_sp = it->second;
        g_driver.storage_registry.erase(it);
    }

    std::lock_guard<std::mutex> guard(storage_sp->lock);
    storage_sp->closing = true;
    /* Release each device's reference to its Handle individually. */
    for (auto& device : storage_sp->devices)
        handle_release(device.handle_sp.get());
    /* Remove all device entries. 
     * Since these entries hold `shared_ptr<HandleState>` objects, 
     * removing them also releases the corresponding shared ownership references. */
    storage_sp->devices.clear();
    storage_sp->capacity_bytes = 0;
}
