#include "runtime/execution/dispatcher.h"

namespace runtime{
    Status Dispatcher::Dispatch(const DispatchKey& key, ExecutionContext& context){
        auto kernel = registry_.Find(key);
        if(kernel == nullptr){
            return Status::Error(StatusCode::NotFound, "Kernel not found.");
        }
        return kernel->Execute(context);
    }

    Status Dispatcher::Register(const DispatchKey& key, std::shared_ptr<Kernel> kernel){
        return registry_.Register(key, std::move(kernel));
    }

    std::shared_ptr<Kernel> Dispatcher::Find(const DispatchKey& key){
        return registry_.Find(key);
    }
}