#pragma once

#include <functional>

#include "runtime/operators/operator.h"
#include "runtime/tensor/tensor.h"

namespace runtime{
    class AddOperator final : public Operator{
        public:
            
            AddOperator() = delete;
            AddOperator(const Tensor& A, const Tensor& B, Tensor& C);
            
            Status Execute(Dispatcher& dispatcher) override; 
            OperatorType type() const noexcept override;

        private:
            std::reference_wrapper<const Tensor> A_;
            std::reference_wrapper<const Tensor> B_;
            std::reference_wrapper<Tensor> C_;
    };
}