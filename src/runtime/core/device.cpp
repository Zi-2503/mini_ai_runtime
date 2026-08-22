#include "runtime/core/device.h"

namespace runtime{

    // ==================================
    // 1. Initializer
    // ==================================

    Device::Device() noexcept = default;

    Device::Device(DeviceType type) noexcept
        : type_(type),
          index_(0)
    {}
    Device::Device(DeviceType type, int index) noexcept
        : type_(type),
          index_(index)
    {}

    // ==================================
    // 2. Observer
    // ==================================
    DeviceType Device::type()  const noexcept { return type_;  }
    int        Device::index() const noexcept { return index_; }

    // ==================================
    // 3. Comparison
    // ==================================
    bool Device::operator==(const Device& other) const noexcept{
        return type_ == other.type_ && index_ == other.index_;
    }
    bool Device::operator!=(const Device& other) const noexcept{
        return !(*this == other);
    }

}

