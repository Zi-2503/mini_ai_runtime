#include "runtime/core/shape.h"

#include <numeric>
#include <cassert>

namespace runtime{

    // ==========================================
    // 1. Initialize
    // ==========================================

    Shape::Shape(std::initializer_list<size_t> dims)
        : dims_(dims)
    {}
    Shape::Shape(std::vector<size_t> dims)
        : dims_(std::move(dims))
    {}

    // ==========================================
    // 2. Query
    // ==========================================

    size_t Shape::rank() const noexcept{
        return dims_.size();
    }
    size_t Shape::numel() const noexcept{
        return std::accumulate(dims_.begin(), dims_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    }
    const size_t& Shape::operator[](size_t dim) const noexcept{
        return dims_[dim];
    }
    size_t& Shape::operator[](size_t dim) noexcept{
        return dims_[dim];
    }

    // ==========================================
    // 3. Iterator
    // ==========================================

    auto Shape::begin() const noexcept{
        return dims_.begin();
    }
    auto Shape::end() const noexcept{
        return dims_.end();
    }

    // ==========================================
    // 4. Comparison
    // ==========================================

    bool Shape::operator==(const Shape& other) const noexcept{
        return dims_ == other.dims_;
    }

    bool Shape::operator!=(const Shape& other) const noexcept{
        return !(*this == other);
    }

    // ==========================================
    // 5. Linear to index
    // ==========================================

    std::vector<size_t>Shape::linear_to_index(size_t linear) const{
        assert(linear < numel());
        std::vector<size_t> indices(rank());
        for (size_t i = rank(); i-- > 0; ){
            indices[i] = linear % dims_[i];
            linear /= dims_[i];
        }
        return indices;
    }

}