#include "runtime/core/dtype.h"

namespace runtime{

    size_t SizeOf(DataType dtype) noexcept{
        switch(dtype){
            case DataType::Bool:        return sizeof(bool);
            case DataType::Int32:       return sizeof(int32_t);
            case DataType::Int64:       return sizeof(int64_t);
            case DataType::Float32:     return sizeof(float);
            case DataType::Float64:     return sizeof(double);
        }
        return 0;
    }

    std::string_view ToString(DataType dtype) noexcept{
        switch(dtype){
            case DataType::Bool:        return "Bool";
            case DataType::Int32:       return "Int32";
            case DataType::Int64:       return "Int64";
            case DataType::Float32:     return "Float32";
            case DataType::Float64:     return "Float64";
        }
        return "Unknown";
    }

}