#include <cassert>
#include <iostream>

#include "runtime/execution/dispatcher.h"
#include "runtime/operators/add_operator.h"
#include "runtime/kernel/cpu/cpu_add_kernel.h"

using namespace runtime;

void testBasicRuntime();
void testWrongDataType();
void testKernelNotFound();

int main(){

    testBasicRuntime();
    testWrongDataType();
    testKernelNotFound();

}

void testBasicRuntime(){
    Dispatcher dispatcher;
    dispatcher.Register(
        DispatchKey(OperatorType::Add, Device(DeviceType::CPU), DataType::Float32),
        std::make_shared<CPUAddKernel<float>>()
    );

    Tensor A(Shape({3}), DataType::Float32, Device(DeviceType::CPU));
    Tensor B(Shape({3}), DataType::Float32, Device(DeviceType::CPU));
    Tensor C(Shape({3}), DataType::Float32, Device(DeviceType::CPU));
    
    auto* a = A.data<float>();
    auto* b = B.data<float>();

    for(int i = 0; i < 3; ++i){
        a[i] = float(i);
        b[i] = float(3 - i);
    }

    AddOperator add(A, B, C);
    Status status = add.Execute(dispatcher);

    assert(status.ok());

    auto* c = C.data<float>();
    for(int i = 0; i < 3; ++i){
        assert(c[i] == float(3));
    }

    std::cout << "[PASS] Runtime\n";

}

void testWrongDataType()
{
    Dispatcher dispatcher;

    Tensor A(Shape({2, 3}), DataType::Bool, Device(DeviceType::CPU));
    Tensor B(Shape({2, 3}), DataType::Bool, Device(DeviceType::CPU));
    Tensor C(Shape({2, 3}), DataType::Bool, Device(DeviceType::CPU));

    AddOperator add(A, B, C);
    Status status = add.Execute(dispatcher);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Invalid Datatype\n";
}

void testKernelNotFound(){

    Dispatcher dispatcher;
    ExecutionContext context;

    Status status = dispatcher.Dispatch(
        DispatchKey(OperatorType::Add, Device(DeviceType::CPU), DataType::Bool),
        context
    );

    assert(!status.ok());
    assert(status.code() == StatusCode::NotFound);

    std::cout << "[PASS] Kernel not Found\n";
}