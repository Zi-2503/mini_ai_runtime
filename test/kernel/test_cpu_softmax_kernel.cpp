#include "runtime/kernel/cpu/cpu_softmax_kernel.h"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace runtime;

void testBasicSoftmax();
void testNonContiguousSoftmax();
void testNumericalStability();

int main(){

    testBasicSoftmax();
    testNonContiguousSoftmax();
    testNumericalStability();

}

void testBasicSoftmax(){

    // ========================================
    // 1. Create input
    // ========================================

    Tensor A(Shape({2, 3}),DataType::Float32);

    float* a = A.data<float>();

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    a[3] = 1.0f;
    a[4] = 1.0f;
    a[5] = 1.0f;


    // ========================================
    // 2. Create output
    // ========================================

    Tensor B(Shape({2, 3}),DataType::Float32);

    // ========================================
    // 3. Execute
    // ========================================

    CPUSoftmaxKernel<float> kernel;

    ExecutionContext context;

    context.AddInput(A);
    context.AddOutput(B);

    Status status = kernel.Execute(context);

    assert(status.ok());


    // ========================================
    // 4. Reference
    // ========================================

    const float e1 = std::exp(1.0f);
    const float e2 = std::exp(2.0f);
    const float e3 = std::exp(3.0f);

    const float sum = e1 + e2 + e3;

    const float ref0 = e1 / sum;
    const float ref1 = e2 / sum;
    const float ref2 = e3 / sum;

    const float ref3 = 1.0f / 3.0f;


    // ========================================
    // 5. Check
    // ========================================

    const float* b = B.data<float>();

    constexpr float tol = 1e-5f;

    assert(std::abs(b[0] - ref0) < tol);
    assert(std::abs(b[1] - ref1) < tol);
    assert(std::abs(b[2] - ref2) < tol);

    assert(std::abs(b[3] - ref3) < tol);
    assert(std::abs(b[4] - ref3) < tol);
    assert(std::abs(b[5] - ref3) < tol);


    // ========================================
    // 6. Row sum
    // ========================================

    assert(std::abs(b[0] + b[1] + b[2] - 1.0f) < tol);
    assert(std::abs(b[3] + b[4] + b[5] - 1.0f) < tol);


    std::cout << "[PASS] Basic Softmax\n";

}


void testNonContiguousSoftmax() {

    // ========================================
    // 1. Create input
    // ========================================

    Tensor A(Shape({2, 3}), DataType::Float32);

    float* a = A.data<float>();

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;

    a[3] = 4.0f;
    a[4] = 5.0f;
    a[5] = 6.0f;


    // ========================================
    // 2. Transpose
    // ========================================

    Tensor B = A.transpose(0, 1);

    // B =
    //
    // [1 4]
    // [2 5]
    // [3 6]
    //
    // B is non-contiguous.

    // ========================================
    // 3. Output
    // ========================================

    Tensor C(Shape({3, 2}), DataType::Float32);

    // ========================================
    // 4. Execute
    // ========================================

    CPUSoftmaxKernel<float> kernel;

    ExecutionContext context;

    context.AddInput(B);
    context.AddOutput(C);

    Status status = kernel.Execute(context);

    assert(status.ok());

    // ========================================
    // 5. Reference
    // ========================================

    float* c = C.data<float>();

    constexpr float tol = 1e-5f;

    for (size_t row = 0; row < 3; ++row) {

        const float x0 = static_cast<float>(row + 1);
        const float x1 = static_cast<float>(row + 4);
        const float e0 = std::exp(x0);
        const float e1 = std::exp(x1);
        const float sum = e0 + e1;
        const float ref0 = e0 / sum;
        const float ref1 = e1 / sum;

        assert(std::abs(c[row * 2] - ref0) < tol);
        assert(std::abs(c[row * 2 + 1] - ref1) < tol);

        // Row sum should be 1.
        assert(std::abs(c[row * 2] + c[row * 2 + 1] - 1.0f) < tol);
    }

    std::cout << "[PASS] NonContiguous Softmax\n";
}

void testNumericalStability() {

    // ========================================
    // 1. Create input
    // ========================================

    Tensor A(Shape({2, 3}), DataType::Float32);

    float* a = A.data<float>();

    a[0] = 1000.0f;
    a[1] = 1001.0f;
    a[2] = 1002.0f;

    a[3] = 2000.0f;
    a[4] = 2001.0f;
    a[5] = 2002.0f;


    // ========================================
    // 2. Output
    // ========================================

    Tensor B(Shape({2, 3}), DataType::Float32);


    // ========================================
    // 3. Execute
    // ========================================

    CPUSoftmaxKernel<float> kernel;

    ExecutionContext context;

    context.AddInput(A);
    context.AddOutput(B);

    Status status = kernel.Execute(context);

    assert(status.ok());


    // ========================================
    // 4. Check finite
    // ========================================

    const float* b = B.data<float>();

    for (size_t i = 0; i < 6; ++i) {
        assert(std::isfinite(b[i]));
        assert(b[i] >= 0.0f);
    }

    // ========================================
    // 5. Reference
    // ========================================

    // softmax([1000,1001,1002]) == softmax([-2,-1,0])

    const float e0 = std::exp(-2.0f);
    const float e1 = std::exp(-1.0f);
    const float e2 = std::exp(0.0f);
    const float sum = e0 + e1 + e2;
    const float ref0 = e0 / sum;
    const float ref1 = e1 / sum;
    const float ref2 = e2 / sum;

    constexpr float tol = 1e-5f;

    // ========================================
    // 6. Rows
    // ========================================

    assert(std::abs(b[0] - ref0) < tol);
    assert(std::abs(b[1] - ref1) < tol);
    assert(std::abs(b[2] - ref2) < tol);
    assert(std::abs(b[3] - ref0) < tol);
    assert(std::abs(b[4] - ref1) < tol);
    assert(std::abs(b[5] - ref2) < tol);

    // ========================================
    // 8. Row sums
    // ========================================

    assert(std::abs(b[0] + b[1] + b[2] - 1.0f) < tol);
    assert(std::abs(b[3] + b[4] + b[5] - 1.0f) < tol);

    std::cout << "[PASS] Numerical Stablity\n";
}