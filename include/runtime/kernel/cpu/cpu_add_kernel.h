#pragma once

#include "runtime/kernel/kernel.h"
#include "runtime/core/dtype.h"
#include "runtime/core/status.h"

namespace runtime{
    template<typename T>
    class CPUAddKernel final : public Kernel{

        static_assert(DataTypeTraits<T>::value != DataType::Bool, "CPUAddKernel does not support bool.");

        public:
            CPUAddKernel() = default;
            ~CPUAddKernel() override = default;

            Status Execute(ExecutionContext& context) override;

            [[nodiscard]] std::string_view name() const noexcept override {
                return "CPUAddKernel";
            }
    };
}

#include "runtime/kernel/cpu/cpu_add_kernel.tpp"