#include <cassert>
#include <iostream>

#include "runtime/kernel/cpu/cpu_matmul_kernel.h"

using namespace runtime;

void TestConstruction();
void TestBasicMatMul();
void TestZeroMatrix();
void TestIdentityMatrix();

void TestWrongInputNumber();
void TestWrongOutputNumber();

void TestInputShapeMismatch();
void TestOutputShapeMismatch();

void TestDtypeMismatch();

void TestNonContiguousA();
void TestNonContiguousB();
void TestNonContiguousAB();

int main(){

    TestConstruction();
    TestBasicMatMul();
    TestZeroMatrix();
    TestIdentityMatrix();

    TestWrongInputNumber();
    TestWrongOutputNumber();

    TestInputShapeMismatch();
    TestOutputShapeMismatch();

    TestDtypeMismatch();

    TestNonContiguousA();
    TestNonContiguousB();
    TestNonContiguousAB();

}

void TestConstruction(){
    CPUMatMulKernel<float> kernel;
    std::cout << "[PASS] Construction\n";
}

void TestBasicMatMul(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({3,4}), DataType::Float32);
    Tensor C(Shape({2,4}), DataType::Float32);
    
    float* a = A.data<float>();
    float* b = B.data<float>();

    for(size_t i = 0; i < 6;  ++i)      a[i] = i + 1;
    for(size_t i = 0; i < 12; ++i)      b[i] = i + 1;
    
    ExecutionContext ctx;
    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUMatMulKernel<float> kernel;
    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* c = C.data<float>();

    assert(c[0] == 38 && c[1] == 44 && c[2] == 50 && c[3] == 56);
    assert(c[4] == 83 && c[5] == 98 && c[6] == 113 && c[7] == 128);

    std::cout << "[PASS] Basic MatMul\n";

}

void TestZeroMatrix(){

    Tensor A(Shape({2,2}), DataType::Float32);
    Tensor B(Shape({2,2}), DataType::Float32);
    Tensor C(Shape({2,2}), DataType::Float32);

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

    CPUMatMulKernel<float> kernel;
    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* c = C.data<float>();

    for(size_t i = 0; i < 4; ++i){
        assert(c[i] == 0.0f);
    }

    std::cout << "[PASS] ZeroMatrix\n";

}

void TestIdentityMatrix(){

    Tensor A(Shape({2,2}), DataType::Float32);
    Tensor B(Shape({2,2}), DataType::Float32);
    Tensor C(Shape({2,2}), DataType::Float32);

    float* a = A.data<float>();
    float* b = B.data<float>();

    
    for(size_t i = 0; i < 4; ++i)   a[i] = 0.0, b[i] = float(i);
    for(size_t i = 0; i < 2; ++i)   a[i + 2*i] = 1.0f;

    ExecutionContext ctx;
    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUMatMulKernel<float> kernel;
    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* c = C.data<float>();

    for(size_t i = 0; i < 4; ++i){
        assert(c[i] == b[i]);
    }

    std::cout << "[PASS] IdentityMatrix\n";

}

void TestWrongInputNumber(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,2}), DataType::Float32);

    CPUMatMulKernel<float> kernel;
    ExecutionContext ctx;

    ctx.AddInput(A);
    ctx.AddOutput(C);

    Status status = kernel.Execute(ctx);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Wrong Input Number\n";

}


void TestWrongOutputNumber(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);
    Tensor D(Shape({2,3}), DataType::Float32);

    CPUMatMulKernel<float> kernel;
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
    Tensor B(Shape({2,3}), DataType::Float32);
    Tensor C(Shape({2,3}), DataType::Float32);

    CPUMatMulKernel<float> kernel;
    ExecutionContext ctx;

    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    Status status = kernel.Execute(ctx);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Input Shape Mismatch\n";

}


void TestOutputShapeMismatch(){
    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({3,3}), DataType::Float32);
    Tensor C(Shape({2,4}), DataType::Float32);

    CPUMatMulKernel<float> kernel;
    ExecutionContext ctx;

    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    Status status = kernel.Execute(ctx);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Output Shape Mismatch\n";
}

void TestDtypeMismatch(){

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B(Shape({3,3}), DataType::Bool);
    Tensor C(Shape({2,3}), DataType::Float32);    

    CPUMatMulKernel<float> kernel;
    ExecutionContext context;

    context.AddInputs({A, B});
    context.AddOutput(C);

    Status status = kernel.Execute(context);

    assert(!status.ok());
    assert(status.code() == StatusCode::InvalidArgument);

    std::cout << "[PASS] Input Datatype Mismatch\n";

}

void TestNonContiguousA(){

    // ============================================
    // A_base : [3, 2]
    // A      : [2, 3]  <-- transpose view, non-contiguous
    //
    // B      : [3, 4]
    // C      : [2, 4]
    // ============================================

    Tensor A_base(Shape({3,2}), DataType::Float32);
    Tensor B(Shape({3,4}), DataType::Float32);
    Tensor C(Shape({2,4}), DataType::Float32);

    float* a_base = A_base.data<float>();
    float* b = B.data<float>();

    // A_base =
    //
    // [1, 2]
    // [3, 4]
    // [5, 6]
    //
    for(size_t i = 0; i < 6; ++i)
        a_base[i] = static_cast<float>(i + 1);

    // B =
    //
    // [ 1,  2,  3,  4]
    // [ 5,  6,  7,  8]
    // [ 9, 10, 11, 12]
    //
    for(size_t i = 0; i < 12; ++i)
        b[i] = static_cast<float>(i + 1);

    // A =
    //
    // [1, 3, 5]
    // [2, 4, 6]
    //
    Tensor A = A_base.transpose(0, 1);

    // Expected:
    //
    // A @ B =
    //
    // [1,3,5] @ B
    //
    // = [ 1*1 + 3*5 + 5*9,
    //     1*2 + 3*6 + 5*10,
    //     1*3 + 3*7 + 5*11,
    //     1*4 + 3*8 + 5*12 ]
    //
    // = [61, 70, 79, 88]
    //
    //
    // [2,4,6] @ B
    //
    // = [ 2*1 + 4*5 + 6*9,
    //     2*2 + 4*6 + 6*10,
    //     2*3 + 4*7 + 6*11,
    //     2*4 + 4*8 + 6*12 ]
    //
    // = [76, 88, 100, 112]

    ExecutionContext ctx;

    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUMatMulKernel<float> kernel;

    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* c = C.data<float>();

    assert(c[0] == 61);
    assert(c[1] == 70);
    assert(c[2] == 79);
    assert(c[3] == 88);

    assert(c[4] == 76);
    assert(c[5] == 88);
    assert(c[6] == 100);
    assert(c[7] == 112);

    std::cout << "[PASS] NonContiguous A MatMul\n";
}

void TestNonContiguousB(){

    // ============================================
    // A : [2,3]
    // B_base : [4,3]
    // B : [3,4] <-- non-contiguous
    // C : [2,4]
    // ============================================

    Tensor A(Shape({2,3}), DataType::Float32);
    Tensor B_base(Shape({4,3}), DataType::Float32);
    Tensor C(Shape({2,4}), DataType::Float32);

    float* a = A.data<float>();
    float* b_base = B_base.data<float>();

    // A =
    //
    // [1, 2, 3]
    // [4, 5, 6]
    //
    for(size_t i = 0; i < 6; ++i)
        a[i] = static_cast<float>(i + 1);

    // B_base =
    //
    // [ 1,  2,  3]
    // [ 4,  5,  6]
    // [ 7,  8,  9]
    // [10, 11, 12]
    //
    for(size_t i = 0; i < 12; ++i)
        b_base[i] = static_cast<float>(i + 1);

    // B = B_base^T
    //
    // [ 1,  4,  7, 10]
    // [ 2,  5,  8, 11]
    // [ 3,  6,  9, 12]
    //
    Tensor B = B_base.transpose(0, 1);

    ExecutionContext ctx;

    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUMatMulKernel<float> kernel;

    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* c = C.data<float>();

    // Expected:
    //
    // [1,2,3] @ B
    //
    // [ 1*1 + 2*2 + 3*3,
    //   1*4 + 2*5 + 3*6,
    //   1*7 + 2*8 + 3*9,
    //   1*10 + 2*11 + 3*12 ]
    //
    // = [14, 32, 50, 68]
    //
    //
    // [4,5,6] @ B
    //
    // = [32, 77, 122, 167]

    assert(c[0] == 14);
    assert(c[1] == 32);
    assert(c[2] == 50);
    assert(c[3] == 68);

    assert(c[4] == 32);
    assert(c[5] == 77);
    assert(c[6] == 122);
    assert(c[7] == 167);

    std::cout << "[PASS] NonContiguous B MatMul\n";
}

void TestNonContiguousAB(){

    // ============================================
    // A_base : [3,2]
    // A      : [2,3] <-- non-contiguous
    //
    // B_base : [4,3]
    // B      : [3,4] <-- non-contiguous
    //
    // C      : [2,4]
    // ============================================

    Tensor A_base(Shape({3,2}), DataType::Float32);
    Tensor B_base(Shape({4,3}), DataType::Float32);
    Tensor C(Shape({2,4}), DataType::Float32);

    float* a_base = A_base.data<float>();
    float* b_base = B_base.data<float>();

    // A_base =
    //
    // [1, 2]
    // [3, 4]
    // [5, 6]
    //
    for(size_t i = 0; i < 6; ++i)
        a_base[i] = static_cast<float>(i + 1);

    // B_base =
    //
    // [ 1,  2,  3]
    // [ 4,  5,  6]
    // [ 7,  8,  9]
    // [10, 11, 12]
    //
    for(size_t i = 0; i < 12; ++i)
        b_base[i] = static_cast<float>(i + 1);

    // A = A_base^T
    //
    // [1, 3, 5]
    // [2, 4, 6]
    //
    Tensor A = A_base.transpose(0, 1);

    // B = B_base^T
    //
    // [ 1,  4,  7, 10]
    // [ 2,  5,  8, 11]
    // [ 3,  6,  9, 12]
    //
    Tensor B = B_base.transpose(0, 1);

    ExecutionContext ctx;

    ctx.AddInput(A);
    ctx.AddInput(B);
    ctx.AddOutput(C);

    CPUMatMulKernel<float> kernel;

    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* c = C.data<float>();

    // Expected:
    //
    // [1,3,5] @ B
    //
    // [ 1*1 + 3*2 + 5*3,
    //   1*4 + 3*5 + 5*6,
    //   1*7 + 3*8 + 5*9,
    //   1*10 + 3*11 + 5*12 ]
    //
    // = [22, 49, 76, 103]
    //
    //
    // [2,4,6] @ B
    //
    // = [28, 64, 100, 136]

    assert(c[0] == 22);
    assert(c[1] == 49);
    assert(c[2] == 76);
    assert(c[3] == 103);

    assert(c[4] == 28);
    assert(c[5] == 64);
    assert(c[6] == 100);
    assert(c[7] == 136);

    std::cout << "[PASS] NonContiguous A&B MatMul\n";
}