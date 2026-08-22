#include <cassert>
#include <iostream>

#include "runtime/kernel/cpu/cpu_add_kernel.h"

using namespace runtime;

void TestConstruction();
void TestBasicAdd();
void TestZeroTensor();
void TestNegativeValues();

void TestWrongInputNumber();
void TestWrongOutputNumber();

void TestInputShapeMismatch();
void TestOutputShapeMismatch();

void TestDtypeMismatch();

void TestNonContiguousTensor();

int main(){

    TestConstruction();
    TestBasicAdd();
    TestZeroTensor();
    TestNegativeValues();

    TestWrongInputNumber();
    TestWrongOutputNumber();

    TestInputShapeMismatch();
    TestOutputShapeMismatch();

    TestDtypeMismatch();

    TestNonContiguousTensor();

}

void TestConstruction(){
    CPUAddKernel<float> kernel;
    std::cout << "[PASS] Construction\n";
}

void TestBasicAdd(){
    
    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);

    float* a = A.data<float>();
    float* b = B.data<float>();
    
    for(size_t i = 0; i < 6; ++i){
        a[i] = float(i + 1);
        b[i] = float(6 - i);
    }

    ExecutionContext ctx;
    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUAddKernel<float> kernel;
    Status status = kernel.Execute(ctx);

    assert(status.ok());

    const float* c = C.data<float>();

    for(size_t i = 0; i < 6; ++i){
        assert(c[i] == 7.0);
    }

    std::cout << "[PASS] Basic Add\n";

}

void TestZeroTensor(){

    Tensor A(Shape({4}), DataType::Float32);
    Tensor B(Shape({4}), DataType::Float32);
    Tensor C(Shape({4}), DataType::Float32);

    float* a = A.data<float>();
    float* b = B.data<float>();

    for(size_t i = 0; i < 4; ++i){
        a[i] = 0.0f;
        b[i] = float(i);
    }

    ExecutionContext ctx;
    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUAddKernel<float> kernel;
    Status status = kernel.Execute(ctx);
    
    assert(status.ok());

    float* c = C.data<float>();
    for(size_t i = 0; i < 4; i++){
        assert(c[i] == float(i));
    }

    std::cout << "[PASS] Zero Tensor\n";

}


void TestNegativeValues(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);

    float* a = A.data<float>();
    float* b = B.data<float>();
    
    for(int i = 0; i < 6; ++i){
        a[i] = float( i + 1);
        b[i] = float(-i - 1);
    }

    ExecutionContext ctx;
    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUAddKernel<float> kernel;
    Status status = kernel.Execute(ctx);

    assert(status.ok());

    const float* c = C.data<float>();

    for(size_t i = 0; i < 6; ++i){
        assert(c[i] == 0.0);
    }

    std::cout << "[PASS] Negative Values\n";
}

void TestWrongInputNumber(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);

    CPUAddKernel<float> kernel;
    ExecutionContext context;

    context.AddInput(A);
    context.AddOutput(C);

    Status status = kernel.Execute(context);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Wrong Input Number\n";

}

void TestWrongOutputNumber(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);
    Tensor D(Shape({2,3}), DataType::Float32);

    CPUAddKernel<float> kernel;
    ExecutionContext context;

    context.AddInput(A);
    context.AddInput(B);
    context.AddOutput(C);

    context.AddOutput(D);

    Status status = kernel.Execute(context);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Wrong Output Number\n";

}

void TestInputShapeMismatch(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({3,2}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);    

    CPUAddKernel<float> kernel;
    ExecutionContext context;

    context.AddInput(A);
    context.AddInput(B);
    
    context.AddOutput(C);

    Status status = kernel.Execute(context);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Input Shape Mismatch\n";

}

void TestOutputShapeMismatch(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({3,2}), DataType::Float32);    

    CPUAddKernel<float> kernel;
    ExecutionContext context;

    context.AddInput(A);
    context.AddInput(B);
    
    context.AddOutput(C);

    Status status = kernel.Execute(context);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Output Shape Mismatch\n";

}

void TestDtypeMismatch(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({2,3}), DataType::Bool);
    Tensor C(Shape({2,3}), DataType::Float32);    

    CPUAddKernel<float> kernel;
    ExecutionContext context;

    context.AddInput(A);
    context.AddInput(B);
    
    context.AddOutput(C);

    Status status = kernel.Execute(context);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Input Datatype Mismatch\n";

}


void TestNonContiguousTensor(){

    Tensor A(Shape({2,3}), DataType::Float32);
    float* a = A.data<float>();

    Tensor B(Shape({3,2}), DataType::Float32);
    float* b = B.data<float>();

    for(int i = 0; i < 6; ++i){
        a[i] = float(i);
        b[i] = float(i);
    }

    // A = [0, 1, 2]
    //     [3, 4, 5]
    // B = [0, 1]
    //     [2, 3]
    //     [4, 5]
    // C = [0, 2, 4]
    //     [1, 3, 5]
    // D should be [0, 3, 6]
    //             [4, 7, 10]

    Tensor C = B.transpose(0, 1);
    Tensor D(Shape({2,3}), DataType::Float32);

    CPUAddKernel<float> kernel;

    ExecutionContext ctx;
    ctx.AddInputs({A,C});
    ctx.AddOutput(D);

    Status status = kernel.Execute(ctx);

    assert(status.ok());
    
    float* d = D.data<float>();
    assert(d[0] == 0);
    assert(d[1] == 3);
    assert(d[2] == 6);
    assert(d[3] == 4);
    assert(d[4] == 7);
    assert(d[5] == 10);

    std::cout << "[PASS] NonContiguous Tensor Add\n";
}