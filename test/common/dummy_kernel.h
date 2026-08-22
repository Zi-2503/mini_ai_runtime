#pragma once 

#include "runtime/kernel/kernel.h"
#include "runtime/core/dtype.h"
#include "runtime/core/status.h"
#include "runtime/execution/execution_context.h"

namespace runtime::testing{

    class DummyKernel final : public Kernel{
        public:
            DummyKernel() = default;
            ~DummyKernel() = default;

            std::string_view name() const noexcept override {
                return "DummyKernel";
            }
            Status Execute(ExecutionContext&) override{
                executed_ = true;
                executed_count_++;
                return Status::OK();
            }
            [[nodiscard]]bool executed() const noexcept{
                return executed_;
            }
            [[nodiscard]]size_t executed_count() const noexcept{
                return executed_count_;
            } 


        private:
            bool executed_ = false;
            size_t executed_count_ = 0;
    };

}