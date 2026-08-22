#include "runtime/iterator/reduction/reduction_iterator_item.h"
#include "runtime/tensor/tensor.h"
#include "runtime/core/stride.h"

namespace runtime{

    ReductionIteratorItem::ReductionIteratorItem(
        const Tensor& input,
        Tensor& output,
        size_t input_offset,
        size_t output_offset
    ) :
        input_(input),
        output_(output),
        input_offset_(input_offset),
        output_offset_(output_offset)
    {}

    size_t ReductionIteratorItem::size() const noexcept{
        return input_.dim(input_.rank() - 1);
    }

    const Stride ReductionIteratorItem::input_stride() const noexcept{
        return input_.stride();
    }

    const Stride ReductionIteratorItem::output_stride() const noexcept{
        return output_.stride();
    }


}