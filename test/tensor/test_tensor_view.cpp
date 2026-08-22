#include "runtime/tensor/tensor.h"
#include "runtime/core/dtype.h"
#include "runtime/core/device.h"

#include <cstring>
#include <iostream>
#include <cassert>

using namespace runtime;

int main(){

    Tensor A(Shape({2,3}), DataType::Float32);
    float value[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    memcpy(A.data<float>(), value, sizeof(value));

    Tensor B = A.Create_View(A, Shape({3,2}), Stride({1,3}), 0);
    
    assert(A.storage().get() == B.storage().get());
    assert(A.shape() != B.shape());
    assert(A.stride() != B.stride());
    assert(A.dtype() == B.dtype());
    assert(A.device() == B.device());
    assert(A.is_contiguous() && !B.is_contiguous());

    std::cout << "[PASS] Create View\n";

    Tensor C = A.transpose(0,1);

    assert(A.storage().get() == C.storage().get());
    assert(A.shape()[0] == C.shape()[1]);
    assert(A.shape()[1] == C.shape()[0]);
    assert(A.stride()[0] == C.stride()[1]);
    assert(A.stride()[1] == C.stride()[0]);
    assert(A.device() == C.device() && A.offset() == C.offset() && A.dtype() == C.dtype());
    assert(A.is_contiguous() && !(C.is_contiguous()));
    assert(A.at<float>(0,1) == C.at<float>(1,0));

    std::cout << "[PASS] Transpose\n";

}