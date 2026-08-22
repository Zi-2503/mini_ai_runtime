#include "runtime/core/stride.h"

namespace runtime{
    
    Stride::Stride(std::initializer_list<size_t> strides)
        : strides_(strides)
    {}
    
    Stride::Stride(std::vector<size_t> strides)
        : strides_(std::move(strides))
    {}

    const size_t& Stride::operator[](size_t index) const noexcept{
        return strides_[index];
    }

    size_t& Stride::operator[](size_t index) noexcept{
        return strides_[index];
    }

    size_t Stride::rank() const noexcept{
        return strides_.size();
    }

    bool Stride::operator==(const Stride& other) const noexcept{
        return strides_ == other.strides_;
    }

    bool Stride::operator!=(const Stride& other) const noexcept{
        return !(*this == other); 
    }

}