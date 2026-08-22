#pragma once

#include "runtime/kernel/kernel.h"
#include "runtime/core/dtype.h"
#include "runtime/core/status.h"

namespace runtime{
    template<typename T>
    class CPUMatMulKernel final : public Kernel{

        static_assert(DataTypeTraits<T>::value != DataType::Bool, "CPUMatMulKernel does not support bool.");

        public:
            CPUMatMulKernel() = default;
            ~CPUMatMulKernel() override = default;

            Status Execute(ExecutionContext& context) override;

            [[nodiscard]] std::string_view name() const noexcept override{
                return "CPUMatMulKernel";
            }
    };
}

#include "runtime/kernel/cpu/cpu_matmul_kernel.tpp"