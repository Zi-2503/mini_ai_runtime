#include "runtime/memory/allocator.h"

#include <cstdlib>

namespace runtime{

    void* Allocator::Allocate(size_t bytes) const{
        return std::malloc(bytes);
    }

    void Allocator::Free(void* ptr) const{
        std::free(ptr);
    }

}