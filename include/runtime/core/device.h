// device.h
#pragma once

#include <functional>

#include "runtime/core/hash.h"

namespace runtime{
    enum class DeviceType{
        CPU,
        CUDA
    };

    class Device{
        public:
            Device() noexcept;
            explicit Device(DeviceType type)    noexcept;
            Device(DeviceType type, int index)  noexcept;

            [[nodiscard]]DeviceType type()  const noexcept;
            [[nodiscard]]int index()        const noexcept;
            
            bool operator==(const Device&) const noexcept;
            bool operator!=(const Device&) const noexcept;

        private:
            DeviceType type_;
            int index_;
    };
}

namespace std {

    template<>
    struct hash<runtime::Device>{
        size_t operator()(const runtime::Device& device) const noexcept{
            size_t seed = 0;
            runtime::HashCombine(seed, std::hash<int>{}(static_cast<int>(device.type())));
            runtime::HashCombine(seed, std::hash<int>{}(device.index()));
            return seed;
        }
    };

} // namespace std