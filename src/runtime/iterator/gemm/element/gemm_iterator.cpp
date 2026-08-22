#include "runtime/iterator/gemm/element/gemm_iterator.h"

namespace runtime{

    GEMMIterator::GEMMIterator(GEMMProblem& problem)
        : problem_(&problem)
    {}

    size_t GEMMIterator::num_elements() const noexcept{
        return problem_->M() * problem_->N();
    }

    GEMMIterator::Iterator GEMMIterator::begin(){
        return Iterator(this, 0);
    }

    GEMMIterator::Iterator GEMMIterator::end(){
        return Iterator(this, num_elements());
    }

    GEMMIteratorItem GEMMIterator::item(size_t linear) const{
        const size_t N = problem_->N();
        const size_t row = linear / N;
        const size_t col = linear % N;
        return GEMMIteratorItem(problem_->A(), problem_->B(), problem_->C(), row, col);
    }

    GEMMIterator::Iterator::Iterator(GEMMIterator* iterator,size_t index)
        : iterator_(iterator),
          index_(index) 
    {}


    GEMMIterator::Iterator& GEMMIterator::Iterator::operator++() {
        ++index_;
        return *this;
    }


    bool GEMMIterator::Iterator::operator!=(const Iterator& other) const {
        return index_ != other.index_;
    }


    GEMMIteratorItem GEMMIterator::Iterator::operator*() {
        return iterator_->item(index_);
    }

}