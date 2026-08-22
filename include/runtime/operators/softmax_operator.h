#pragma once

#include <functional>

#include "runtime/operators/operator.h"
#include "runtime/tensor/tensor.h"

namespace runtime{
    class SoftmaxOperator final : public Operator{
        public:
            
            SoftmaxOperator() = delete;
            SoftmaxOperator(const Tensor& A, Tensor& B);
            
            Status Execute(Dispatcher& dispatcher) override; 
            OperatorType type() const noexcept override;

        private:
            std::reference_wrapper<const Tensor> A_;
            std::reference_wrapper<Tensor> B_;
    };
}
