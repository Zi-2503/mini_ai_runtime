#include <stdexcept>
#include <vector>

#include "runtime/iterator/reduction/reduction_iterator.h"

namespace runtime{

    ReductionIterator::ReductionIterator(const Tensor& input, Tensor& output)
        : input_(&input),
          output_(&output)
    {
        const size_t rank = input.rank();
        if(rank == 0){
            throw std::invalid_argument("Reduction Iterator: Requires Tensor has at least 1 dimension");
        }

        std::vector<size_t> dims;
        dims.reserve(rank - 1);
        for(size_t i = 0; i + 1 < rank; ++i){
            dims.push_back(input.dim(i));
        }
        outer_shape_ = Shape(dims);
    }

    ReductionIterator::Iterator::Iterator(ReductionIterator* iterator, size_t index)
        : iterator_(iterator),
          index_(index)
    {}

    ReductionIterator::Iterator& ReductionIterator::Iterator::operator++(){
        ++index_;
        return *this;
    }

    bool ReductionIterator::Iterator::operator!=(const ReductionIterator::Iterator& other) const{
        return index_ != other.index_;
    }

    ReductionIteratorItem ReductionIterator::Iterator::operator*(){
        return iterator_ -> item(index_);
    }

    ReductionIterator::Iterator ReductionIterator::begin(){
        return Iterator(this, 0);
    }

    ReductionIterator::Iterator ReductionIterator::end(){
        return Iterator(this, num_groups());
    }

    size_t ReductionIterator::num_groups() const noexcept{
        return outer_shape_.numel();
    }

    ReductionIteratorItem ReductionIterator::item(size_t linear){
        auto outer_indices = outer_shape_.linear_to_index(linear);
        outer_indices.push_back(0);
        const size_t input_offset  = input_ -> element_offset(outer_indices.data());
        const size_t output_offset = output_-> element_offset(outer_indices.data());

        return ReductionIteratorItem(*input_, *output_, input_offset, output_offset);
    } 

}