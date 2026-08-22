#pragma once

#include <cstddef>
#include <vector>

#include "runtime/tensor/tensor.h"

namespace runtime {

    class ElementwiseIteratorItem {

        public:

            ElementwiseIteratorItem(
                const std::vector<const Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                std::vector<size_t> input_offsets, std::vector<size_t> output_offsets
            );

            // ====================================
            // 1. Input
            // ====================================

            template<typename T>
            [[nodiscard]] const T* input(size_t index) const{
                return reinterpret_cast<const T*>(inputs_[index]->raw_data<T>()) + input_offsets_[index];
            }

            // ====================================
            // 2. Output
            // ====================================

            template<typename T>
            [[nodiscard]] T* output(size_t index){
                return reinterpret_cast<T*>(outputs_[index]->raw_data<T>()) + output_offsets_[index];
            }

        private:

            std::vector<const Tensor*> inputs_;
            std::vector<Tensor*> outputs_;

            std::vector<size_t> input_offsets_;
            std::vector<size_t> output_offsets_;

    };

}