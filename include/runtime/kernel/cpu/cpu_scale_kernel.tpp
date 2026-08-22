#pragma once

#include "runtime/core/status.h"
#include "runtime/tensor/tensor.h"
#include "runtime/execution/execution_context.h"
#include "runtime/kernel/cpu/cpu_scale_kernel.h"

namespace runtime{
    template<typename T> Status CPUScaleKernel<T>::Execute(ExecutionContext& context){

        // ================================
        // 1. Validate context
        // ================================

        if(context.inputs().size() != 1)    return Status::Error(StatusCode::InvalidArgument, "CPUScaleKernel requires 1 input tensor.");
        if(context.outputs().size() != 1)   return Status::Error(StatusCode::InvalidArgument, "CPUScaleKernel requires 1 output tensor");

        // ================================
        // 2. Bind tensor & Validate
        // ================================

        const Tensor& A = context.inputs()[0];
        Tensor& B = context.outputs()[0];

        if (A.dtype() != DataTypeTraits<T>::value || B.dtype() != DataTypeTraits<T>::value) {
            return Status::Error(StatusCode::InvalidArgument, "All tensors must have the same data type as the kernel template parameter");
        }   

        if(A.rank() != B.rank())                  return Status::Error(StatusCode::InvalidArgument, "Input and output tensor rank mismatch.");
        for(size_t i = 0; i < A.rank(); ++i){
            if(A.dim(i) != B.dim(i))              return Status::Error(StatusCode::InvalidArgument, "Input and output tensor shape mismatch.");
        }

        // ================================
        // 3. Get scale factor
        // ================================

        if(!context.HasAttribute("scale"))  return Status::Error(StatusCode::InvalidArgument, "Scale attribute is missing.");
        const float scale_factor = context.GetAttribute<float>("scale");

        // ================================
        // 4. Compute
        // ================================

        auto iterator = context.elementwise_iterator();

        for (auto item : iterator) {
            const T* a = item.template input<T>(0);
            T* b = item.template output<T>(0);

            *b = *a * static_cast<T>(scale_factor);
        }
        return Status::OK();
    
    }
}