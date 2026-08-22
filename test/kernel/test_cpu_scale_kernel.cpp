#include "runtime/kernel/cpu/cpu_scale_kernel.h"

#include <iostream>
#include <cassert>

using namespace runtime;

void TestBasicScale();
void TestNonContiguousScale();

int main(){

    TestBasicScale();
    TestNonContiguousScale();

}

void TestBasicScale(){

    Tensor A(Shape({2,3}), DataType::Float32);
    float* a = A.data<float>();

    for(int i = 0; i < 6; ++i){
        a[i] = float(i);
    }

    float scale = 0.5;

    Tensor B(Shape({2,3}), DataType::Float32);

    ExecutionContext ctx;
    ctx.AddInput(A);
    ctx.AddOutput(B);
    ctx.SetAttribute("scale", scale);

    CPUScaleKernel<float> kernel;
    Status status = kernel.Execute(ctx);

    assert(status.ok());

    float* b = B.data<float>();
    for(int i = 0; i < 6; ++i){
        assert(b[i] == float(i) * 0.5);
    }

    std::cout << "[PASS] Basic scale\n";

}

void TestNonContiguousScale(){

    Tensor A(Shape({2,3}), DataType::Float32);
    float* a = A.data<float>();
    for(int i = 0; i < 6; ++i){
        a[i] = float(i);
    }
    Tensor B = A.transpose(0, 1);
    float scale = 0.5;

    Tensor C(Shape({3,2}), DataType::Float32);

    ExecutionContext ctx;
    ctx.AddInput(B);
    ctx.AddOutput(C);
    ctx.SetAttribute("scale", scale);

    CPUScaleKernel<float> kernel;

    Status status = kernel.Execute(ctx);

    float* c = C.data<float>();
    float expected[] = {0.0f, 1.5f, 0.5f, 2.0f, 1.0f, 2.5f};
    for (int i = 0; i < 6; ++i) {
        assert(c[i] == expected[i]);
    }

    std::cout << "[PASS] Noncontiguous Scale\n";

}

