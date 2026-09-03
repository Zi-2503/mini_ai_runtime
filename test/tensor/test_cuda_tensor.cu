#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#include "runtime/tensor/tensor.h"

using namespace runtime;

void testCUDATensor();

int main(){

    testCUDATensor();

}

void testCUDATensor(){

    Tensor A(Shape({1024}), DataType::Int32, DeviceType::CUDA);

    assert(A.rank() == 2);
    assert(A.device() == DeviceType::CUDA);
    assert(A.dtype() == DataType::Int32);
    assert(A.bytes() == 1024 * sizeof(int));
    assert(A.empty() == false);

    std::cout << "[PASS] CUDA Tensor" << std::endl;

}