#pragma once

#include <cstddef>

#include "runtime/tensor/tensor.h"

namespace runtime{

    class GEMMIteratorItem{
        
        public:
            
            GEMMIteratorItem(const Tensor& A, const Tensor& B, Tensor& C, size_t row, size_t col);

            [[nodiscard]] size_t row() const noexcept;
            [[nodiscard]] size_t col() const noexcept;

            [[nodiscard]] size_t K() const noexcept;

            [[nodiscard]] size_t a_stride() const noexcept;
            [[nodiscard]] size_t b_stride() const noexcept;

            template<typename T> 
            [[nodiscard]] const T* a() const noexcept{
                const size_t offset = row_ * A_.stride()[0];
                return reinterpret_cast<const T*>(A_.raw_data<T>()) + offset;
            }

            template<typename T>
            [[nodiscard]] const T* b() const noexcept{
                const size_t offset = col_ * B_.stride()[1];
                return reinterpret_cast<const T*>(B_.raw_data<T>()) + offset;
            }

            template<typename T>
            [[nodiscard]] T* c() const noexcept{
                const size_t offset = row_ * C_.stride()[0] + col_ * C_.stride()[1];
                return reinterpret_cast<T*>(C_.raw_data<T>()) + offset;
            }

        private:
            const Tensor& A_;
            const Tensor& B_;
            Tensor& C_;

            size_t row_;
            size_t col_;
    };

}