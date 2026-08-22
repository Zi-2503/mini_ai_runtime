#include "runtime/iterator/elementwise/elementwise_iterator_item.h"

namespace runtime{

    ElementwiseIteratorItem::ElementwiseIteratorItem(
        const std::vector<const Tensor*>& inputs, const std::vector<Tensor*>& outputs,
        std::vector<size_t> input_offsets, std::vector<size_t> output_offsets
    )
        : inputs_(std::move(inputs)),
          outputs_(std::move(outputs)),
          input_offsets_(std::move(input_offsets)),
          output_offsets_(std::move(output_offsets))
    {}


}