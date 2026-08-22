#pragma once

#include "runtime/core/status.h"
#include "runtime/execution/operator_type.h"
// #include "runtime/execution/dispatcher.h"

namespace runtime{

    class Dispatcher;

    class Operator{
        public:
            virtual ~Operator() = default;
            virtual Status Execute(Dispatcher& Dispatcher) = 0;
            virtual OperatorType type() const noexcept = 0;
    };

}