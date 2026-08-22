//operator_type.h
#pragma once

namespace runtime{

    enum class OperatorType{
        Add,
        MatMul,
        Attention,
        Softmax,
        Scale
    };

}