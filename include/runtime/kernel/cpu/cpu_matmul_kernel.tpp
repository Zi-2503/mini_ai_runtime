#pragma once

#include "runtime/core/status.h"
#include "runtime/tensor/tensor.h"
#include "runtime/execution/execution_context.h"
#include "runtime/kernel/cpu/cpu_matmul_kernel.h"

#include "runtime/iterator/gemm/element/gemm_iterator.h"
#include "runtime/iterator/gemm/problem/gemm_problem.h"

namespace runtime{

    template<typename T>
    Status CPUMatMulKernel<T>::Execute(ExecutionContext& context){

        // ====================================
        // 1. Validate context
        // ====================================

        if(context.inputs().size()  != 2)   return Status::Error(StatusCode::InvalidArgument, "CPUMatMulKernel excepts 2 input tensors.");
        if(context.outputs().size() != 1)   return Status::Error(StatusCode::InvalidArgument, "CPUMatMulKernel excepts 1 output tensor.");

        // ====================================
        // 2. Bind Tensor
        // ====================================

        const Tensor& A = context.inputs()[0].get();
        const Tensor& B = context.inputs()[1].get();
        Tensor& C = context.outputs()[0].get();

        if (A.dtype() != DataTypeTraits<T>::value || B.dtype() != DataTypeTraits<T>::value || C.dtype() != DataTypeTraits<T>::value) {
            return Status::Error(StatusCode::InvalidArgument, "All tensors must have the same data type as the kernel template parameter");
        }    
        
        // ====================================
        // 3. Validate Shape
        // ====================================

        if(A.rank() != 2 || B.rank() != 2 || C.rank() != 2)     return Status::Error(StatusCode::InvalidArgument, "CPUMatMulKernel except rank 2 tensors.");

        size_t M = A.dim(0), K = A.dim(1);
        if(B.dim(0) != K)   return Status::Error(StatusCode::InvalidArgument, "Input tensors dimension mismatch.");

        size_t N = B.dim(1);
        if(C.dim(0) != M || C.dim(1) != N)  return Status::Error(StatusCode::InvalidArgument, "Output Tensor dimension mismatch.");

        // ====================================
        // 4. Compute
        // ====================================

        GEMMProblem problem(A, B, C);
        GEMMIterator iterator(problem);

        for(auto item : iterator){
            
            const T* a = item.a<T>();
            const T* b = item.b<T>();
            T* c = item.c<T>();
            T sum = T(0);

            for(size_t k = 0; k < item.K(); ++k){
                sum += a[k * item.a_stride()] * b[k * item.b_stride()];
            }

            *c = sum;

        }


        // ====================================
        // 5. Return 
        // ====================================
        return Status::OK();


    }

}