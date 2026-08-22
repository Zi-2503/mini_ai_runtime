#include "runtime/operators/matmul_operator.h"
#include "runtime/execution/execution_context.h"
#include "runtime/execution/dispatch_key.h"
#include "runtime/execution/dispatcher.h"

namespace runtime{

    MatMulOperator::MatMulOperator(const Tensor& A, const Tensor& B, Tensor& C)
        : A_(A), B_(B), C_(C)
    {}

    OperatorType MatMulOperator::type() const noexcept{
        return OperatorType::MatMul;
    }

    Status MatMulOperator::Execute(Dispatcher& dispatcher){

        // Validate
        const Tensor& A = A_.get();
        const Tensor& B = B_.get();
        const Tensor& C = C_.get();

        if(A.empty() || B.empty() || C.empty())           return Status::Error( StatusCode::InvalidArgument, "MatMulOperator requires valid tensors.");

        size_t M = A.dim(0), N = A.dim(1);
        if(B.dim(0) != N)                   return Status::Error(StatusCode::InvalidArgument, "MatMul tensors dimension mismatch.");

        size_t K = B.dim(1);
        if(C.dim(0) != M && C.dim(1) != K)  return Status::Error(StatusCode::InvalidArgument, "MatMul tensors dimension mismatch.");

        if(A.dtype() == DataType::Bool || B.dtype() == DataType::Bool)  return Status::Error(StatusCode::InvalidArgument, "MatMul does not support bool tensors.");
        if(A.dtype() != B.dtype())      return Status::Error(StatusCode::InvalidArgument, "MatMul requires input tensors to have same dtype.");
        if(C.dtype() != A.dtype())      return Status::Error(StatusCode::InvalidArgument, "MatMul requires output dtype to match input dtype.");

        ExecutionContext context;
        
        context.AddInput (A_);
        context.AddInput (B_);
        context.AddOutput(C_);

        DispatchKey key(type(), A_.get().device(), A_.get().dtype());

        return dispatcher.Dispatch(key, context);
    }    


}