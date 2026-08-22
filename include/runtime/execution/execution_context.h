#pragma once

#include <functional>
#include <vector>

#include "runtime/tensor/tensor.h"
#include "runtime/execution/attribute.h"
#include "runtime/iterator/elementwise/elementwise_iterator.h"

namespace runtime{

    using ExecutionInput  = std::reference_wrapper<const Tensor>;
    using ExecutionOutput = std::reference_wrapper<Tensor>;

    using ExecutionInputs  = std::vector<ExecutionInput>;
    using ExecutionOutputs = std::vector<ExecutionOutput>;

    class ExecutionContext{
        public:
            // ====================================
            // 1. Constructors
            // ====================================

            ExecutionContext() = default;
            ExecutionContext(ExecutionInputs inputs, ExecutionOutputs outputs);

            // ====================================
            // 2. Input
            // ====================================

            [[nodiscard]]const ExecutionInputs& inputs() const noexcept;
            void AddInput(const Tensor& tensor);        
            void AddInputs(std::initializer_list<std::reference_wrapper<const Tensor>> tensors);

            // ====================================
            // 3. Outputs
            // ====================================

            [[nodiscard]]const ExecutionOutputs& outputs() const noexcept;
            void AddOutput(Tensor& tensor);
            void AddOutputs(std::initializer_list<std::reference_wrapper<Tensor>> tensors);

            // ====================================
            // 4. Attributes
            // ====================================

            template<typename T> void SetAttribute(const std::string& name, T value){
                attributes_[name] = Attribute(std::move(value));
            }
            
            template<typename T> T GetAttribute(const std::string& name) const{
                auto it = attributes_.find(name);
                if (it == attributes_.end()) {
                    throw std::out_of_range("ExecutionContext: attribute not found: " + name);
                }
                return it->second.get<T>();
            }
            
            bool HasAttribute(const std::string& name) const noexcept;

            // ====================================
            // 5. Iterator
            // ====================================

            [[nodiscard]]ElementwiseIterator elementwise_iterator() const;

        private:
            ExecutionInputs inputs_;
            ExecutionOutputs outputs_;
            std::unordered_map<std::string, Attribute> attributes_;

    };

}