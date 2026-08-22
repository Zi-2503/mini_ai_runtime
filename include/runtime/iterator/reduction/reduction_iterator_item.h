#pragma once

#include <cstddef>
#include "runtime/tensor/tensor.h"

namespace runtime{

    class ReductionIteratorItem{
        public:
            // ===========================
            // 1. Constructor
            // ===========================
            ReductionIteratorItem(
                const Tensor& input,
                Tensor& output,
                size_t input_offset,
                size_t output_offset
            );

            // ===========================
            // 2. Query
            // ===========================
            [[nodiscard]] size_t size()   const noexcept;
            [[nodiscard]] const Stride input_stride()   const noexcept;
            [[nodiscard]] const Stride output_stride()  const noexcept;

            // ===========================
            // 3. Data access
            // ===========================
            template<typename T>
            const T* input() const{
                return reinterpret_cast<const T*>(input_.raw_data<T>()) + input_offset_;
            }

            template<typename T>
            T* output(){
                return reinterpret_cast<T*>(output_.raw_data<T>()) + output_offset_;
            }

        private:
            const Tensor& input_;
            Tensor& output_;

            size_t input_offset_;
            size_t output_offset_;
    };

}