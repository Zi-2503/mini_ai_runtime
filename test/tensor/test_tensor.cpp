#include <cassert>
#include <cstring>

#include "runtime/tensor/tensor.h"
#include "runtime/core/stride.h"

int main(){

    runtime::Tensor A(runtime::Shape({2,3}), runtime::DataType::Float32);
    float value[] =
    {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };

    std::memcpy(A.data<float>(), value, sizeof(value));

    assert(A.numel() == 6);
    assert(A.stride() == runtime::Stride({3,1}));
    assert(A.dtype() == runtime::DataType::Float32);
    assert(A.shape().rank() == 2);
    assert(A.offset() == 0);
    assert(A.at<float>(1,2) == 6.f);

    runtime::Tensor B;
    assert(B.offset() == 0);

}