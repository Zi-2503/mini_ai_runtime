#include "runtime/execution/execution_context.h"

namespace runtime{

    // ================================
    // 1. Initializer
    // ================================

    ExecutionContext::ExecutionContext(
        ExecutionInputs inputs,
        ExecutionOutputs outputs
    )
        : inputs_(std::move(inputs)),
          outputs_(std::move(outputs))
    {}

    // =================================
    // 2. Inputs & Outputs
    // =================================

    const ExecutionInputs&  ExecutionContext::inputs()  const noexcept { return inputs_;  }
    const ExecutionOutputs& ExecutionContext::outputs() const noexcept { return outputs_; }
    void  ExecutionContext::AddInput(const Tensor& tensor)  { inputs_.emplace_back(std::cref(tensor)); }
    void  ExecutionContext::AddOutput(Tensor& tensor)       { outputs_.emplace_back(std::ref(tensor)); }

    void ExecutionContext::AddInputs(std::initializer_list<std::reference_wrapper<const Tensor>> tensors){
        inputs_.insert(inputs_.end(), tensors.begin(), tensors.end());
    }
    void ExecutionContext::AddOutputs(std::initializer_list<std::reference_wrapper<Tensor>> tensors){
        outputs_.insert(outputs_.end(), tensors.begin(), tensors.end());
    }

    // ====================================
    // 4. Attributes
    // ====================================

    bool ExecutionContext::HasAttribute(const std::string& name) const noexcept{
        return attributes_.find(name) != attributes_.end();
    }

    // ====================================
    // 5. Iterator
    // ====================================

    ElementwiseIterator ExecutionContext::elementwise_iterator() const{
        std::vector<const Tensor*> inputs;
        inputs.reserve(inputs_.size());

        for (const auto& input : inputs_) {
            inputs.push_back(&input.get());
        }

        std::vector<Tensor*> outputs;
        outputs.reserve(outputs_.size());

        for (const auto& output : outputs_) {
            outputs.push_back(&output.get());
        }

        return ElementwiseIterator(inputs, outputs);
    }

}