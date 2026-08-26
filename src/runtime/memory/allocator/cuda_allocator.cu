#include "runtime/memory/allocator/cuda_allocator.h"

#include <cuda_runtime.h>

namespace runtime{
    void* CUDAAllocator::Allocate(std::size_t bytes) const {
        void* ptr = nullptr;
        cudaError_t error = cudaMalloc(&ptr, bytes);
        if (error != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void CUDAAllocator::Free(void* ptr) const {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
}