#pragma once

#include <cstddef>

#include "runtime/tensor/tensor.h"

namespace runtime{

    class GEMMProblem{

        public:
            GEMMProblem(const Tensor& A, const Tensor& B, Tensor& C);

            [[nodiscard]] const Tensor& A() const noexcept;
            [[nodiscard]] const Tensor& B() const noexcept;
            [[nodiscard]] Tensor& C() const noexcept;

            [[nodiscard]] size_t M() const noexcept;
            [[nodiscard]] size_t N() const noexcept;
            [[nodiscard]] size_t K() const noexcept;

        private:
            const Tensor* A_;
            const Tensor* B_;
            Tensor* C_;
    };

}