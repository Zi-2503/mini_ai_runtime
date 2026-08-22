// =================================================
// [Warning] Reduction dimension : LAST DIMENSION
// =================================================

#pragma once

#include "runtime/tensor/tensor.h"
#include "runtime/iterator/reduction/reduction_iterator_item.h"

#include <cstddef>
#include <vector>

namespace runtime{

    class ReductionIterator{

        public:

            ReductionIterator(const Tensor& input, Tensor& output);

            class Iterator{
                public:
                
                    Iterator(ReductionIterator* iterator, size_t index);

                    Iterator& operator++();
                    bool operator!=(const Iterator& other) const;
                    ReductionIteratorItem operator*();
                
                private:
                    ReductionIterator* iterator_;
                    size_t index_;
            };

            Iterator begin();
            Iterator end();

        private:
            
            ReductionIteratorItem item(size_t linear);
            
            [[nodiscard]] size_t num_groups() const noexcept;

        private:
            const Tensor* input_;
            Tensor* output_;

            Shape outer_shape_;
    };

}
