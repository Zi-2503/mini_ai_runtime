#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>

namespace runtime{

    class Stride{
        public:

            Stride() = default;
            explicit Stride(std::initializer_list<size_t> strides);
            explicit Stride(std::vector<size_t> strides);

            [[nodiscard]] const size_t& operator[](size_t i) const noexcept;
            [[nodiscard]] size_t& operator[](size_t i) noexcept;
            [[nodiscard]] size_t rank() const noexcept;


            bool operator==(const Stride&) const noexcept;
            bool operator!=(const Stride&) const noexcept;

        private:
            std::vector<size_t> strides_;
    };
}