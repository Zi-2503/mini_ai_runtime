#include "runtime/operators/scale_operator.h"
#include "runtime/execution/execution_context.h"
#include "runtime/execution/dispatch_key.h"
#include "runtime/execution/dispatcher.h"

namespace runtime{

    ScaleOperator::ScaleOperator(const Tensor& A, Tensor& B, float scale_factor)
        : A_(A), B_(B), scale_factor_(scale_factor)
    {}

    OperatorType ScaleOperator::type() const noexcept{
        return OperatorType::Scale;
    }

    Status ScaleOperator::Execute(Dispatcher& dispatcher){

        const Tensor& A = A_.get();
        Tensor& B = B_.get();

        if(A.empty() || B.empty())      return Status::Error(StatusCode::InvalidArgument, "ScaleOperator requires valid tensors.");
        if(A.shape() != B.shape())      return Status::Error(StatusCode::InvalidArgument, "ScaleOperator requires input and output tensors to have same shape.");
        if(A.dtype() != B.dtype())      return Status::Error(StatusCode::InvalidArgument, "ScaleOperator requires input and output tensors to have same dtype.");
        if(A.device() != B.device())    return Status::Error(StatusCode::InvalidArgument, "ScaleOperator requires input and output tensors to be on same device.");

        if(A.dtype() == DataType::Bool) return Status::Error(StatusCode::InvalidArgument, "ScaleOperator does not support bool tensors.");
        if(A.dtype() == DataType::Int32 || A.dtype() == DataType::Int64) return Status::Error(StatusCode::InvalidArgument, "ScaleOperator does not support int32/int64 tensors.");

        ExecutionContext context;
        context.AddInput (A_);
        context.AddOutput(B_);
        context.SetAttribute<float>("scale", scale_factor_);

        DispatchKey key(type(), A.device(), A.dtype());

        return dispatcher.Dispatch(key, context);

    }

}