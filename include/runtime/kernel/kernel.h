#pragma once

#include <string_view>

#include "runtime/core/status.h"
#include "runtime/execution/execution_context.h"

namespace runtime{
    class Kernel{
        public:
            
            virtual ~Kernel() = default;

            Kernel() = default;

            Kernel(const Kernel&) = delete;
            Kernel& operator=(const Kernel&) = delete;

            Kernel(Kernel&&) = delete;
            Kernel& operator=(Kernel&&) = delete;

            virtual Status Execute(ExecutionContext& context) = 0;
            virtual std::string_view name() const noexcept = 0;
    };
}