#include "runtime/iterator/gemm/element/gemm_iterator_item.h"

namespace runtime{

    GEMMIteratorItem::GEMMIteratorItem(const Tensor& A, const Tensor& B, Tensor& C, size_t row, size_t col)
        : A_(A),
          B_(B),
          C_(C),
          row_(row),
          col_(col)
    {}

    size_t GEMMIteratorItem::row() const noexcept{
        return row_;
    }

    size_t GEMMIteratorItem::col() const noexcept{
        return col_;
    }

    size_t GEMMIteratorItem::K() const noexcept{
        return A_.dim(1);
    }

    size_t GEMMIteratorItem::a_stride() const noexcept{
        return A_.stride()[1];
    }

    size_t GEMMIteratorItem::b_stride() const noexcept{
        return B_.stride()[0];
    }

}