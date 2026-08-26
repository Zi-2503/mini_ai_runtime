#include "runtime/kernel/cuda/cuda_add_kernel.h"

#include <cuda_runtime.h>

namespace runtime{

    template<typename T>
    __global__ void vector_add(const T* A, const T* B, T* C, size_t N){
        int i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(i < N){
            C[i] = A[i] + B[i];
        }
    }

    template<typename T>
    __global__ void CUDAAddKernel<T>::Execute(ExecutionContext& ctx){
        
    }


}