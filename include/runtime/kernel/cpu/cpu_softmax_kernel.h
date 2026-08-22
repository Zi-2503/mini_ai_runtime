#pragma once

#include "runtime/kernel/kernel.h"
#include "runtime/core/dtype.h"
#include "runtime/core/status.h"

namespace runtime{
    template<typename T>
    class CPUSoftmaxKernel final : public Kernel{

        static_assert(
            DataTypeTraits<T>::value != DataType::Bool || 
            DataTypeTraits<T>::value != DataType::Int32 ||
            DataTypeTraits<T>::value != DataType::Int64,
            "CPUSoftmaxKernel only support float/double."
        );

        public:
            CPUSoftmaxKernel() = default;
            ~CPUSoftmaxKernel() = default;

            Status Execute(ExecutionContext& context) override;

            [[nodiscard]] std::string_view name() const noexcept override{
                return "CPUSoftmaxKernel";
            }

    };
}

#include "runtime/kernel/cpu/cpu_softmax_kernel.tpp"