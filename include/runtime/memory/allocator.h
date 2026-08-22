#pragma once

#include <stddef.h>

namespace runtime{

    /**
     * @brief 
     * Allocates and releases raw memory.
     * Allocator is responsible only for acquiring and releasing memory from the underlying system.
     * It does NOT own the allocated memory. Memory ownership belongs to Storage.
    **/

    class Allocator{

        public:
            Allocator()  = default;
            ~Allocator() = default;
            
            /**
             * @brief Allocate a contiguous memory block.
             * @param bytes Number of bytes to allocate.
             * @return Pointer to the beginning of the allocated memory.
             */
            [[nodiscard]]
            void* Allocate(size_t bytes) const;

            /**
             * @brief Release a previously allocated memory block.
             * @param ptr Pointer returned by Allocate().
             */
            void Free(void* ptr) const;

    };

}