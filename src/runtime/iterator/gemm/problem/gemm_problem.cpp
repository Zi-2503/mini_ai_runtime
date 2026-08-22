#include <stdexcept>

#include "runtime/iterator/gemm/problem/gemm_problem.h"

namespace runtime{

    GEMMProblem::GEMMProblem(const Tensor& A, const Tensor& B, Tensor& C)
        : A_(&A),
          B_(&B),
          C_(&C)
    {
        if(A.dim(1) != B.dim(0)){
            throw std::invalid_argument("GEMM : input matrices dimension mismatch.");
        }
    }

    size_t GEMMProblem::M() const noexcept{
        return A_->dim(0);
    }
    
    size_t GEMMProblem::K() const noexcept{
        return A_->dim(1);
    }

    size_t GEMMProblem::N() const noexcept{
        return B_->dim(1);
    }

    const Tensor& GEMMProblem::A() const noexcept{
        return *A_;
    }

    const Tensor& GEMMProblem::B() const noexcept{
        return *B_;
    }

    Tensor& GEMMProblem::C() const noexcept{
        return *C_;
    }

}