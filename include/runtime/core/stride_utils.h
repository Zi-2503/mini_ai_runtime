#pragma once

#include "runtime/core/stride.h"
#include "runtime/core/shape.h"

namespace runtime{

    Stride Make_Contiguous_Stride(const Shape& shape);

}