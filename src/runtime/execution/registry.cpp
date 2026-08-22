#include "runtime/execution/registry.h"

namespace runtime{
    Status KernelRegistry::Register(const DispatchKey& key, std::shared_ptr<Kernel> kernel){
        if(kernel == nullptr){
            return Status::Error(StatusCode::InvalidArgument, "Kernel is nullptr.");
        }
        
        auto [it, inserted] = kernels_.emplace(key, std::move(kernel));

        if(!inserted){
            return Status::Error(StatusCode::AlreadyExist, "Kernel is already exist.");
        }
        
        return Status::OK(); 
    }

    std::shared_ptr<Kernel> KernelRegistry::Find(const DispatchKey& key) const{
        auto it = kernels_.find(key);
        if(it == kernels_.end()){
            return nullptr;
        }
        return it->second;
    }
}