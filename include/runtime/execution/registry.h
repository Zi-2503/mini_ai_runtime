#pragma once

#include <memory>
#include <unordered_map>

#include "runtime/core/status.h"
#include "runtime/execution/dispatch_key.h"
#include "runtime/kernel/kernel.h"

namespace runtime{

    class KernelRegistry{

        public:
            KernelRegistry() = default;
            
            Status Register(const DispatchKey& key, std::shared_ptr<Kernel> kernel);

            std::shared_ptr<Kernel> Find(const DispatchKey& key) const;

        private:
            std::unordered_map<DispatchKey, std::shared_ptr<Kernel>> kernels_;
    };

}