#pragma once

#include "runtime/memory/allocator.h"

namespace runtime{

    class CUDAAllocator final : public Allocator{
        public:
            void* Allocate(size_t bytes) const override;
            void Free(void* ptr) const override;
    };    

}