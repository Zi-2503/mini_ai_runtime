#include <cassert>
#include <iostream>
#include <memory>

#include "runtime/execution/dispatch_key.h"
#include "runtime/execution/dispatcher.h"

#include "runtime/core/device.h"
#include "runtime/core/device.h"

#include "runtime/execution/operator_type.h"

#include "common/dummy_kernel.h"

using namespace runtime;
using namespace runtime::testing;

void TestRegister();
void TestFind();
void TestDispatch();
void TestDuplicatedRegister();
void TestKernelNotFound();

int main(){

    TestRegister();
    TestFind();
    TestDispatch();
    TestDuplicatedRegister();
    TestKernelNotFound();

}

void TestRegister(){
    Dispatcher dispatcher;
    auto kernel = std::make_shared<DummyKernel>();
    DispatchKey key(OperatorType::Add, Device(DeviceType::CPU), DataType::Float32);
    Status status = dispatcher.Register(key, kernel);

    assert(status.ok());
    std::cout << "[PASS] Register\n"; 
}

void TestFind(){
    Dispatcher dispatcher;
    auto kernel = std::make_shared<DummyKernel>();
    DispatchKey key(OperatorType::Add, Device(DeviceType::CPU), DataType::Float32);
    Status status = dispatcher.Register(key, kernel);

    assert(dispatcher.Find(key) == kernel);
    std::cout << "[PASS] Find\n";
}

void TestDispatch(){
    Dispatcher dispatcher;
    auto kernel = std::make_shared<DummyKernel>();
    DispatchKey key(OperatorType::Add, Device(DeviceType::CPU), DataType::Float32);
    Status status = dispatcher.Register(key, kernel);

    ExecutionContext context;
    status = dispatcher.Dispatch(key, context);

    assert(status.ok());
    assert(kernel->executed());
    assert(kernel->executed_count() == 1);
    std::cout << "[PASS] Dispatch\n";
}

void TestDuplicatedRegister(){
    Dispatcher dispatcher;
    auto kernel = std::make_shared<DummyKernel>();
    DispatchKey key(OperatorType::Add, Device(DeviceType::CPU), DataType::Float32);
    Status status = dispatcher.Register(key, kernel);
    status = dispatcher.Register(key, kernel);

    assert(!status.ok());
    assert(status.code() == StatusCode::AlreadyExist);
    std::cout << "[PASS] Duplicated Register\n";
}

void TestKernelNotFound(){
    Dispatcher dispatcher;
    auto kernel = std::make_shared<DummyKernel>();
    DispatchKey key(OperatorType::Add, Device(DeviceType::CPU), DataType::Float32);
    Status status = dispatcher.Register(key, kernel);
    
    DispatchKey missing_key(OperatorType::MatMul, Device(DeviceType::CUDA), DataType::Float64);
    ExecutionContext context;
    status = dispatcher.Dispatch(missing_key, context);

    assert(!status.ok());
    assert(status.code() == StatusCode::NotFound);
    std::cout << "[PASS] Kernel not Found\n";
}