#pragma once

#include <functional>

#include "runtime/kernel/kernel.h"
#include "runtime/core/dtype.h"
#include "runtime/core/status.h"

namespace runtime{

    template<typename T>
    class CPUScaleKernel final : public Kernel{

        static_assert(DataTypeTraits<T>::value != DataType::Bool, "CPUScaleKernel does not support bool.");
        static_assert(DataTypeTraits<T>::value != DataType::Int32, "CPUScaleKernel does not support int32.");
        static_assert(DataTypeTraits<T>::value != DataType::Int64, "CPUScaleKernel does not support int64.");

        public:
            CPUScaleKernel() = default;
            ~CPUScaleKernel() override = default;

            Status Execute(ExecutionContext& context) override;

            [[nodiscard]] std::string_view name() const noexcept override {
                return "CPUScaleKernel";
            }
    };

}

#include "runtime/kernel/cpu/cpu_scale_kernel.tpp"