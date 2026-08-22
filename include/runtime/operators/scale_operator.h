#pragma once

#include <functional>

#include "runtime/operators/operator.h"
#include "runtime/tensor/tensor.h"

namespace runtime{
    class ScaleOperator final : public Operator{
        public:
            
            ScaleOperator() = delete;
            ScaleOperator(const Tensor& A, Tensor& B, float scale_factor);
            
            Status Execute(Dispatcher& dispatcher) override; 
            OperatorType type() const noexcept override;

        private:
            std::reference_wrapper<const Tensor> A_;
            std::reference_wrapper<Tensor> B_;
            float scale_factor_;
    };
}
