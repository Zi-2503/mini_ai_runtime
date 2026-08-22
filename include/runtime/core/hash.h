#pragma once

#include <cstddef>

namespace runtime {

    /**
     * @brief Combine a hash value into an existing hash seed.
     *
     * This implementation is adopted from Boost::hash_combine.
     * It provides a simple and effective way to combine multiple hash values into a single hash.
     */
    inline void HashCombine(std::size_t& seed, std::size_t value) noexcept
    {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

}