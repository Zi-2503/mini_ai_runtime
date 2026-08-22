#pragma once

#include <cstddef>
#include "runtime/memory/allocator.h"

namespace runtime{
    /**
     * @brief Owns a contiguous block of raw memory.
     * Storage manages the lifetime of an allocated memory block.
     * It knows nothing about
     *  - tensor shape
     *  - data type
     *  - tensor operations
     * Storage only owns memory.
     */

    class Storage{

        public:
            
            /**
             * @brief Allocate a memory block of the specified size.
             * @param bytes Number of bytes to allocate.
             * @param allocator Allocator used to obtain memory. 
             */
            Storage(size_t bytes, const Allocator& allocator);
            ~Storage();

            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;
            Storage(Storage&&) noexcept = default;
            Storage& operator=(Storage&&) noexcept = default;

            // ==================================
            // Query
            // ==================================

            [[nodiscard]]   void* data() noexcept;
            [[nodiscard]]   const void* data() const noexcept;
            [[nodiscard]]   size_t bytes() noexcept;

        private:

            void* data_;                    // Beginning of the owned memory block
            size_t bytes_;                  // Size of the allocated memory block
            const Allocator& allocator_;    // Allocator responsible for releasing memory

    };

} 