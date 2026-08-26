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

            // version 1.0
            // Allocator()  = default;
            // ~Allocator() = default;
            // void* Allocate(size_t bytes) const;
            // void Free(void* ptr) const;

            // version 2.0
            virtual ~Allocator() = default;

            virtual void* Allocate(size_t bytes) const = 0;

            virtual void Free(void* ptr) const = 0;

    };

}