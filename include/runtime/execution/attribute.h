#pragma once

#include <variant>
#include <vector>
#include <string>

namespace runtime {

    using AttributeValue = std::variant<
        int,
        int64_t,
        float,
        double,
        bool,
        std::string
    >;

    class Attribute{
        public:
            Attribute() = default;
            template<typename T> Attribute(T value)
                : value_(value)
            {}
            template<typename T> T get() const{
                return std::get<T>(value_);
            }

        private:
            AttributeValue value_;
    };


}