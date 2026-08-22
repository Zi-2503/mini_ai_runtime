#pragma once

#include "runtime/core/status.h"
#include "runtime/tensor/tensor.h"
#include "runtime/execution/execution_context.h"
#include "runtime/kernel/cpu/cpu_add_kernel.h"

namespace runtime{

    template<typename T>
    Status CPUAddKernel<T>::Execute(ExecutionContext& context){
        
        // ===================================
        // 1. Validate
        // ===================================

        if(context.inputs().size()  != 2)   return Status::Error(StatusCode::InvalidArgument, "CPUAddKernel excepts 2 input tensors.");
        if(context.outputs().size() != 1)   return Status::Error(StatusCode::InvalidArgument, "CPUAddKernel excepts 1 output tensor.");

        // ===================================
        // 2. Bind Tensors
        // ===================================

        const Tensor& A = context.inputs()[0].get();
        const Tensor& B = context.inputs()[1].get();
        Tensor& C = context.outputs()[0].get();

        if(A.shape() != B.shape())  return Status::Error(StatusCode::InvalidArgument, "Input tensors should have the same size.");
        if(C.shape() != A.shape())  return Status::Error(StatusCode::InvalidArgument, "Output tensor should have the same size with input.");

        if(A.dtype() != DataTypeTraits<T>::value)   return Status::Error(StatusCode::InvalidArgument, "Input tensor A has incorrect data type.");
        if(B.dtype() != DataTypeTraits<T>::value)   return Status::Error(StatusCode::InvalidArgument, "Input tensor B has incorrect data type.");
        if(C.dtype() != DataTypeTraits<T>::value)   return Status::Error(StatusCode::InvalidArgument, "Output tensor has incorrect data type.");

        if(A.empty() || B.empty() || C.empty())     return Status::Error(StatusCode::InvalidArgument, "Tensor has no storage.");

        if(A.device().type() != DeviceType::CPU)    return Status::Error(StatusCode::InvalidArgument, "CPUAddKernel requires CPU Tensors");
        if(A.device() != B.device())                return Status::Error(StatusCode::InvalidArgument, "Input Tensors should have the same device.");
        if(A.device() != C.device())                return Status::Error(StatusCode::InvalidArgument, "Output tensor should have the same device with input.");

        // ===================================
        // 3. Compute
        // ===================================
        auto iterator = context.elementwise_iterator();

        for (auto item : iterator) {
            const T* a = item.template input<T>(0);
            const T* b = item.template input<T>(1);
            T* c = item.template output<T>(0);

            *c = *a + *b;
        }

        // ===================================
        // 4. Return
        // ===================================

        return Status::OK();
    }

}