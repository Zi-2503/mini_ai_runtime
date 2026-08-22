#include "runtime/operators/add_operator.h"
#include "runtime/execution/execution_context.h"
#include "runtime/execution/dispatch_key.h"
#include "runtime/execution/dispatcher.h"

namespace runtime{

    AddOperator::AddOperator(const Tensor& A, const Tensor& B, Tensor& C)
        : A_(A),B_(B),C_(C)
    {}

    OperatorType AddOperator::type() const noexcept{
        return OperatorType::Add;
    }

    Status AddOperator::Execute(Dispatcher& dispatcher){

        // Validate
        const Tensor& A = A_.get();
        const Tensor& B = B_.get();
        const Tensor& C = C_.get();

        if(A.empty() || B.empty() || C.empty())           return Status::Error( StatusCode::InvalidArgument, "AddOperator requires valid tensors.");

        if(A.shape() != B.shape())      return Status::Error(StatusCode::InvalidArgument, "Add requires input tensors to have same shape.");
        if(C.shape() != A.shape())      return Status::Error(StatusCode::InvalidArgument, "Add requires output tensor to have same shape.");

        if(A.dtype() == DataType::Bool || B.dtype() == DataType::Bool)  return Status::Error(StatusCode::InvalidArgument, "Add does not support bool tensors.");
        if(A.dtype() != B.dtype())      return Status::Error(StatusCode::InvalidArgument, "Add requires input tensors to have same dtype.");
        if(C.dtype() != A.dtype())      return Status::Error(StatusCode::InvalidArgument, "Add requires output dtype to match input dtype.");

        ExecutionContext context;
        
        context.AddInput (A_);
        context.AddInput (B_);
        context.AddOutput(C_);

        DispatchKey key(type(), A_.get().device(), A_.get().dtype());

        return dispatcher.Dispatch(key, context);
    }
}