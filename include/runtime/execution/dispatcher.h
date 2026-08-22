#pragma once

#include "runtime/core/status.h"
#include "runtime/execution/execution_context.h"
#include "runtime/execution/dispatch_key.h"
#include "runtime/execution/registry.h"

namespace runtime{
    class Dispatcher{
        public:
            Dispatcher() = default;
            
            Status Register(const DispatchKey& key, std::shared_ptr<Kernel> kernel);
            Status Dispatch(const DispatchKey& key, ExecutionContext& context);
            std::shared_ptr<Kernel> Find(const DispatchKey& key);
    
        private:
            KernelRegistry registry_;
    };
}