#include "runtime/operators/softmax_operator.h"
#include "runtime/execution/execution_context.h"
#include "runtime/execution/dispatch_key.h"
#include "runtime/execution/dispatcher.h"

namespace runtime{

    SoftmaxOperator::SoftmaxOperator(const Tensor& A, Tensor& B)
        : A_(A), B_(B)
    {}

    OperatorType SoftmaxOperator::type() const noexcept{
        return OperatorType::Softmax;
    }

    Status SoftmaxOperator::Execute(Dispatcher& dispatcher){

        // Validate
        const Tensor& A = A_.get();
        const Tensor& B = B_.get();

        if(A.empty() || B.empty())           return Status::Error( StatusCode::InvalidArgument, "SoftmaxOperator requires valid tensors.");

        if(A.rank() != 2 || B.rank() != 2)                  return Status::Error(StatusCode::InvalidArgument, "SoftmaxOperator only supports rank 2 tensors.");
        if(A.dim(0) != B.dim(0) || A.dim(1) != B.dim(1))    return Status::Error(StatusCode::InvalidArgument, "Input and output tensor shape mismatch.");

        if(A.dtype() == DataType::Bool || B.dtype() == DataType::Bool)      return Status::Error(StatusCode::InvalidArgument, "SoftmaxOperator does not support bool tensors.");
        if(A.dtype() == DataType::Int32 || B.dtype() == DataType::Int32)    return Status::Error(StatusCode::InvalidArgument, "SoftmaxOperator does not support int32 tensors.");
        if(A.dtype() == DataType::Int64 || B.dtype() == DataType::Int64)    return Status::Error(StatusCode::InvalidArgument, "SoftmaxOperator does not support int64 tensors.");
        if(A.dtype() != B.dtype())      return Status::Error(StatusCode::InvalidArgument, "SoftmaxOperator requires input and output tensors to have same dtype.");

        ExecutionContext context;
        
        context.AddInput (A_);
        context.AddOutput(B_);

        DispatchKey key(type(), A_.get().device(), A_.get().dtype());

        return dispatcher.Dispatch(key, context);
    }


}