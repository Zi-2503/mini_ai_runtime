#pragma once

#include <functional>

#include "runtime/operators/operator.h"
#include "runtime/tensor/tensor.h"

namespace runtime {

class AttentionOperator final : public Operator {

    public:

        AttentionOperator() = delete;

        AttentionOperator(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O);

        Status Execute(Dispatcher& dispatcher) override;

        OperatorType type() const noexcept override;

    private:

        std::reference_wrapper<const Tensor> Q_;
        std::reference_wrapper<const Tensor> K_;
        std::reference_wrapper<const Tensor> V_;

        std::reference_wrapper<Tensor> O_;
    };

}