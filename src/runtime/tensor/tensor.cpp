#include "runtime/tensor/tensor.h"
#include "runtime/memory/allocator.h"
#include "runtime/core/device.h"

#include "runtime/memory/allocator/cpu_allocator.h"
#include "runtime/memory/allocator/cuda_allocator.h"

#include <algorithm>
#include <stdexcept>
#include <memory>

// namespace{
//     const runtime::Allocator default_allocator;
// }

namespace runtime{

    namespace {

        CPUAllocator cpu_allocator;
        CUDAAllocator cuda_allocator;

        const Allocator& GetAllocator(Device device) {
            switch (device.type()) {
                case DeviceType::CPU:       return cpu_allocator;
                case DeviceType::CUDA:      return cuda_allocator;
                default:                    throw std::runtime_error("Unsupported device.");
            }
        }

    }


    Tensor::Tensor()
        : shape_(),
          dtype_(DataType::Float32),
          storage_(nullptr),
          offset_(0),
          stride_(),
          device_()
        {}

    Tensor::Tensor(const Shape& shape, DataType dtype, const Device device)
        : shape_(std::move(shape)),
          dtype_(dtype),
          offset_(0),
          stride_(std::move(Make_Contiguous_Stride(shape))),
          device_(device)
        {
            const size_t bytes = shape_.numel() * SizeOf(dtype);
            storage_ = std::make_shared<Storage>(bytes, GetAllocator(device));
        }

    Tensor::Tensor(const Shape& shape, DataType dtype, const std::shared_ptr<Storage>& storage, size_t offset, Stride stride, Device device)
        : 
          shape_(std::move(shape)),
          dtype_(dtype),
          storage_(std::move(storage)),
          offset_(offset),
          stride_(std::move(stride)),
          device_(device)
        {}


    // ==========================================
    // Metadata
    // ==========================================

    const Shape& Tensor::shape() const noexcept{
        return shape_;
    }
    
    DataType Tensor::dtype() const noexcept{
        return dtype_;
    } 

    size_t Tensor::numel() const noexcept{
        return shape_.numel();
    }

    size_t Tensor::bytes() const noexcept{
        return storage_ ? storage_ -> bytes() : 0;
    }

    bool Tensor::empty() const noexcept{
        return numel() == 0;
    }

    Device Tensor::device() const noexcept{
        return device_;
    }

    size_t Tensor::rank() const noexcept{
        return shape_.rank();
    }

    size_t Tensor::dim(size_t index) const noexcept{
        return shape_[index];
    }

    size_t Tensor::offset() const noexcept{
        return offset_;
    }

    const Stride& Tensor::stride() const noexcept{
        return stride_;
    }

    const std::shared_ptr<Storage>& Tensor::storage() noexcept{
        return storage_;
    }

    bool Tensor::is_contiguous() const noexcept{
        return stride_ == Make_Contiguous_Stride(shape_);
    }

    bool Tensor::has_storage() const noexcept{
        return !(storage_ == nullptr);
    }

    Tensor Tensor::Create_View(const Tensor& base, Shape shape, Stride stride, size_t offset){
        return Tensor(std::move(shape), base.dtype_, base.storage_, offset, std::move(stride), base.device_);
    }

    Tensor Tensor::transpose(size_t dim0, size_t dim1) const{
        if(dim0 > shape_.rank() || dim1 > shape_.rank()){
            throw std::out_of_range("Transpose dimension out of range.");
        }

        Shape new_shape = shape_;
        Stride new_stride = stride_;

        std::swap(new_shape [dim0], new_shape [dim1]);
        std::swap(new_stride[dim0], new_stride[dim1]);

        return Tensor(new_shape, dtype_, storage_, 0, new_stride, device_);
    }

    size_t Tensor::element_offset(const size_t* indices) const noexcept{
        size_t result = offset_;
        for(size_t i = 0; i < shape_.rank(); ++i){
            result += indices[i] * stride_[i];
        }
        return result;
    }  

}