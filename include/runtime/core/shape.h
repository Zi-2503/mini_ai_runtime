#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>

namespace runtime{

    class Shape{
        public:
            
            // ==================================
            // 1. Construction
            // ==================================

            Shape() = default;                                      // Construct an empty shape
            explicit Shape(std::initializer_list<size_t> dims);     // Construct from an initializer list
            explicit Shape(std::vector<size_t> dims);               // Construct from another vector

            // ==================================
            // 2. Query
            // ==================================

            size_t rank()  const noexcept;                          // Number of tensor dimension
            size_t numel() const noexcept;                          // Total number of tensor elements
            const size_t& operator[](size_t dim) const noexcept;           // Size of the specified dimension
            size_t& operator[](size_t dim) noexcept;
                                                                    // [Warning] No bounds checking is performed here
                                                                    
            // ==================================
            // 3. Iterator
            // ==================================

            auto begin() const noexcept;
            auto end()   const noexcept;

            // ==================================
            // 4. Comparison
            // ==================================

            bool operator==(const Shape&) const noexcept;
            bool operator!=(const Shape&) const noexcept;

            // ==================================
            // 5. Linear to index
            // ==================================
            [[nodiscard]]std::vector<size_t> linear_to_index(size_t linear) const;
        

        private:
            std::vector<size_t> dims_;
    };

}