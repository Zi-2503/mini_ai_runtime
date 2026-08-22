#include "runtime/core/stride_utils.h"

namespace runtime{

    Stride Make_Contiguous_Stride(const Shape& shape){
        size_t rank = shape.rank();
        std::vector<size_t> stride(rank);
        size_t curr = 1;

        for(int i = rank - 1; i >= 0; --i){
            stride[i] = curr;
            curr *= shape[i];
        }

        return Stride(stride);
    }

}