#include "runtime/memory/allocator/cpu_allocator.h"

#include <cstdlib>

namespace runtime{

    void* CPUAllocator::Allocate(size_t bytes) const{
        return std::malloc(bytes);
    }

    void CPUAllocator::Free(void* ptr) const{
        std::free(ptr);
    }

}