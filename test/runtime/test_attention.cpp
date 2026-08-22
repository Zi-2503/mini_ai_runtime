#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

#include "runtime/core/status.h"
#include "runtime/tensor/tensor.h"

#include "runtime/execution/dispatcher.h"

#include "runtime/operators/vanilla_attention_operator.h"

#include "runtime/kernel/cpu/cpu_matmul_kernel.h"
#include "runtime/kernel/cpu/cpu_scale_kernel.h"
#include "runtime/kernel/cpu/cpu_softmax_kernel.h"


using namespace runtime;

void testVanillaAttention();

int main(){

    testVanillaAttention();

}

void testVanillaAttention(){

    Dispatcher dispatcher;
    dispatcher.Register(DispatchKey(OperatorType::MatMul,  Device(DeviceType::CPU), DataType::Float32), std::make_shared<CPUMatMulKernel<float>>());
    dispatcher.Register(DispatchKey(OperatorType::Scale,   Device(DeviceType::CPU), DataType::Float32), std::make_shared<CPUScaleKernel<float>>());
    dispatcher.Register(DispatchKey(OperatorType::Softmax, Device(DeviceType::CPU), DataType::Float32), std::make_shared<CPUSoftmaxKernel<float>>());

    Tensor Q(Shape({2, 1}), DataType::Float32, Device(DeviceType::CPU));
    Tensor K(Shape({2, 1}), DataType::Float32, Device(DeviceType::CPU));
    Tensor V(Shape({2, 2}), DataType::Float32, Device(DeviceType::CPU));
    Tensor O(Shape({2, 2}), DataType::Float32, Device(DeviceType::CPU));

    float* q = Q.data<float>();
    float* k = K.data<float>();
    float* v = V.data<float>();

    q[0] = 2.0f; q[1] = 1.0f;
    k[0] = 1.0f; k[1] = 2.0f;
    v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f; v[3] = 1.0f;

    AttentionOperator attention(Q, K, V, O);
    Status status = attention.Execute(dispatcher);
    assert(status.ok());    
    if (!status.ok()) {
        std::cerr << "[FAIL] Attention: " << status.message() << '\n';
        assert(false);
    }

    float* o = O.data<float>();

    constexpr float expected_00 = 0.1192029f;
    constexpr float expected_01 = 0.8807971f;

    constexpr float expected_10 = 0.2689414f;
    constexpr float expected_11 = 0.7310586f;

    constexpr float tolerance = 1e-5f;

    assert(std::abs(o[0] - expected_00) < tolerance);
    assert(std::abs(o[1] - expected_01) < tolerance);

    assert(std::abs(o[2] - expected_10) < tolerance);
    assert(std::abs(o[3] - expected_11) < tolerance);

    std::cout << "[PASS] Basic Attention\n";

}