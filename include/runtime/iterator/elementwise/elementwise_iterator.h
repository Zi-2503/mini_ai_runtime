#pragma once

#include <cstddef>
#include <vector>

#include "runtime/tensor/tensor.h"
#include "runtime/iterator/elementwise/elementwise_iterator_item.h"

namespace runtime {

    class ElementwiseIterator {

    public:

        // ====================================
        // 1. Constructor
        // ====================================

        ElementwiseIterator(const std::vector<const Tensor*>& inputs, const std::vector<Tensor*>& outputs);

        // ====================================
        // 2. Iterator
        // ====================================

        class Iterator {

            public:

                Iterator(ElementwiseIterator* iter, size_t index);

                Iterator& operator++();
                bool operator!=(const Iterator& other) const;
                ElementwiseIteratorItem operator*();

            private:

                ElementwiseIterator* iter_;
                size_t index_;

        };

        // ====================================
        // 3. Range
        // ====================================

        Iterator begin();
        Iterator end();

    private:

        // ====================================
        // 4. Element access
        // ====================================

        [[nodiscard]] ElementwiseIteratorItem item(size_t linear);

    private:

        std::vector<const Tensor*> inputs_;
        std::vector<Tensor*> outputs_;

        Shape shape_;
    
    };

}