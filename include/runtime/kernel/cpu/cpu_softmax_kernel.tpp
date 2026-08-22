#pragma once

#include "runtime/core/status.h"
#include "runtime/tensor/tensor.h"
#include "runtime/execution/execution_context.h"
#include "runtime/kernel/cpu/cpu_softmax_kernel.h"
#include "runtime/iterator/reduction/reduction_iterator.h"

#include <cmath>

namespace runtime{
 
    template<typename T>
    Status CPUSoftmaxKernel<T>::Execute(ExecutionContext& context){
        
        // ================================
        // 1. Validate context
        // ================================

        if(context.inputs().size() != 1)    return Status::Error(StatusCode::InvalidArgument, "CPUSoftmaxKernel requires 1 input tensor.");
        if(context.outputs().size() != 1)   return Status::Error(StatusCode::InvalidArgument, "CPUSoftmaxKernel requires 1 output tensor");

        // ================================
        // 2. Bind tensor & Validate
        // ================================

        const Tensor& A = context.inputs()[0];
        Tensor& B = context.outputs()[0];

        if (A.dtype() != DataTypeTraits<T>::value || B.dtype() != DataTypeTraits<T>::value) {
            return Status::Error(StatusCode::InvalidArgument, "All tensors must have the same data type as the kernel template parameter");
        }   

        if(A.rank() != 2 || B.rank() != 2)                  return Status::Error(StatusCode::InvalidArgument, "CPUSoftmaxKernel only supports rank 2 tensors.");
        if(A.dim(0) != B.dim(0) || A.dim(1) != B.dim(1))    return Status::Error(StatusCode::InvalidArgument, "Input and output tensor shape mismatch.");

        // =========================
        // 3. Compute
        // =========================
        ReductionIterator iterator(A, B);

        for (auto item : iterator) {

            const float* x = item.input<float>();
            float* y = item.output<float>();
            const size_t n = item.size();
            const size_t sx = item.input_stride()[item.input_stride().rank() - 1];
            const size_t sy = item.output_stride()[item.output_stride().rank() - 1];

            float max_value = x[0];

            for (size_t i = 1; i < n; ++i) {
                max_value = std::max(max_value, x[i * sx]);
            }

            float sum = 0.0f;

            for (size_t i = 0; i < n; ++i) {
                y[i * sy] = std::exp(x[i * sx] - max_value);
                sum += y[i * sy];
            }

            for (size_t i = 0; i < n; ++i) {
                y[i * sy] /= sum;
            }
        }

        return Status::OK();

    }

}