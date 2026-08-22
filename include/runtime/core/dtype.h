#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace runtime{

    /**
     * @brief Runtime representation of a scalar element type.
     * DataType describes how the bytes stored in a tensor should be interpreted.
     * Unlike C++ template types, DataType is determined at runtime.
     */

    enum class DataType{
        Bool,
        Int32,
        Int64,
        Float32,
        Float64
    };

    /**
     * @brief Return the size in bytes of one element.
     */

    [[nodiscard]]
    size_t SizeOf(DataType dtype) noexcept;

    /**
     * @brief Return the human-readable name of a data type.
     */

    [[nodiscard]]
    std::string_view ToString(DataType dtype) noexcept;



    template<typename T>
    struct DataTypeTraits;

    template<> struct DataTypeTraits<float>     {static constexpr DataType value = DataType::Float32;};
    template<> struct DataTypeTraits<double>    {static constexpr DataType value = DataType::Float64;};
    template<> struct DataTypeTraits<int32_t>   {static constexpr DataType value = DataType::Int32;};
    template<> struct DataTypeTraits<int64_t>   {static constexpr DataType value = DataType::Int64;};
    template<> struct DataTypeTraits<bool>      {static constexpr DataType value = DataType::Bool;};
}