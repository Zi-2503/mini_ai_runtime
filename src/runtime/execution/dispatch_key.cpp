#include "runtime/execution/dispatch_key.h"

namespace runtime{

    // ==============================
    // 1. Initializer
    // ==============================

    DispatchKey::DispatchKey(OperatorType op, Device device, DataType dtype) noexcept
        : op_(op),
          device_(device),
          dtype_(dtype)
    {}

    // ==============================
    // 2. Observer
    // ==============================

    OperatorType  DispatchKey::op()         const noexcept  { return op_; }
    const Device& DispatchKey::device()     const noexcept  { return device_; }
    DataType DispatchKey::dtype()           const noexcept  { return dtype_; }

    // ==============================
    // 3. Comparison
    // ==============================
    
    bool DispatchKey::operator==(const DispatchKey& other) const noexcept{
        return op_ == other.op_ && device_ == other.device_ && dtype_ == other.dtype_;
    }

} // namespace runtime

