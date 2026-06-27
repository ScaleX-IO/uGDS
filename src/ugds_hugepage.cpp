#include "ugds_internal.h"
#include <sys/mman.h>

void* hugepage_alloc(size_t size, size_t* alloc_size_out)
{
    size_t alloc_size = (size + UGDS_HUGEPAGE_SIZE - 1) & ~(UGDS_HUGEPAGE_SIZE - 1);

    void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED)
        return nullptr;

    if (alloc_size_out)
        *alloc_size_out = alloc_size;
    return ptr;
}

void hugepage_free(void* ptr, size_t alloc_size)
{
    if (ptr && ptr != MAP_FAILED)
        munmap(ptr, alloc_size);
}
