#pragma once

#include "runtime/iterator/gemm/element/gemm_iterator_item.h"
#include "runtime/iterator/gemm/problem/gemm_problem.h"

namespace runtime{

    class GEMMIterator{

        public:
            
            explicit GEMMIterator(GEMMProblem& problem);

            class Iterator{
                
                public:

                    Iterator(GEMMIterator* iterator, size_t index);

                    Iterator& operator++();
                    bool operator!=(const Iterator& other) const;

                    GEMMIteratorItem operator*();

                private:
                    GEMMIterator* iterator_;
                    size_t index_;

            };
        
            Iterator begin();
            Iterator end();

        private:

            GEMMIteratorItem item(size_t linear) const;
            [[nodiscard]] size_t num_elements() const noexcept;

        private:

            GEMMProblem* problem_;

    };

}