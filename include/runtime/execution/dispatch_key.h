// dispatcher_key.h
#pragma once

#include "functional"

#include "runtime/core/device.h"
#include "runtime/execution/operator_type.h"
#include "runtime/core/dtype.h"
#include "runtime/core/hash.h"

namespace runtime{
    class DispatchKey{
        public:
            DispatchKey(OperatorType op, Device device, DataType dtype) noexcept;

            [[nodiscard]]OperatorType op()          const noexcept;
            [[nodiscard]]const Device& device()     const noexcept;
            [[nodiscard]]DataType dtype()           const noexcept;

            bool operator==(const DispatchKey&) const noexcept;

        private:
            OperatorType op_;
            Device device_;
            DataType dtype_;
    };
}

namespace std {

    template<>
    struct hash<runtime::DispatchKey>{
        size_t operator()(const runtime::DispatchKey& key) const noexcept{
            size_t h = 0;
            runtime::HashCombine(h, std::hash<int>{}(static_cast<int>(key.op())));
            runtime::HashCombine(h, std::hash<int>{}(static_cast<int>(key.device().type())));
            runtime::HashCombine(h, std::hash<int>{}(key.device().index()));
            runtime::HashCombine(h, std::hash<int>{}(static_cast<int>(key.dtype())));
            return h;
        }
    };

} // namespace std