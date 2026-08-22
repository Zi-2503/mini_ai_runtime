#include <stdexcept>

#include "runtime/iterator/elementwise/elementwise_iterator.h"

namespace runtime{

    ElementwiseIterator::ElementwiseIterator(const std::vector<const Tensor*>& inputs, const std::vector<Tensor*>& outputs)
        : inputs_(std::move(inputs)),
          outputs_(std::move(outputs))
    {
        shape_ = outputs_[0]->shape();

        for (const auto* input : inputs_){
            if (input->shape() != shape_){
                throw std::invalid_argument("ElementwiseIterator: input shapes must match");
            }
        }
        for (const auto* output : outputs_){
            if (output->shape() != shape_){
                throw std::invalid_argument("ElementwiseIterator: output shapes must match");
            }
        }
    }

    ElementwiseIterator::Iterator ElementwiseIterator::begin(){
        return Iterator(this, 0);
    }

    ElementwiseIterator::Iterator ElementwiseIterator::end(){
        return Iterator(this, shape_.numel());
    }

    ElementwiseIterator::Iterator::Iterator(ElementwiseIterator* iter, size_t index)
        : iter_(iter),
          index_(index)
    {}

    ElementwiseIterator::Iterator& ElementwiseIterator::Iterator::operator++(){
        index_++;
        return *this;
    }

    bool ElementwiseIterator::Iterator::operator!=(const ElementwiseIterator::Iterator& other) const{
        return index_ != other.index_;
    }

    ElementwiseIteratorItem ElementwiseIterator::Iterator::operator*(){
        return iter_->item(index_);
    }

    ElementwiseIteratorItem ElementwiseIterator::item(size_t linear){
        auto indices = shape_.linear_to_index(linear);

        std::vector<size_t> input_offsets;
        std::vector<size_t> output_offsets;

        input_offsets.reserve(inputs_.size());
        output_offsets.reserve(outputs_.size());

        for(const Tensor* input : inputs_){
            const auto& stride = input->stride();
            size_t offset = 0;
            for(size_t dim = 0; dim < shape_.rank(); ++dim){
                offset += indices[dim] * stride[dim];
            }
            input_offsets.push_back(offset);
        }
        for(const Tensor* output : outputs_){
            const auto& stride = output->stride();
            size_t offset = 0;
            for(size_t dim = 0; dim < shape_.rank(); ++dim){
                offset += indices[dim] * stride[dim];
            }
            output_offsets.push_back(offset);
        }

        return ElementwiseIteratorItem(inputs_, outputs_, std::move(input_offsets), std::move(output_offsets));
    }

}