#pragma once

#include <memory>
#include <cassert>

#include "runtime/core/dtype.h"
#include "runtime/core/shape.h"
#include "runtime/core/device.h"
#include "runtime/core/stride.h"
#include "runtime/core/stride_utils.h"
#include "runtime/memory/storage.h"


namespace runtime{

    /**
     * @brief A typed, multidimensional view over a block of storage.
     * Tensor combines
     *   - Shape
     *   - DataType
     *   - Storage
     *   - Offset
     *   - Stride
     * into a complete runtime object.
     * Tensor does not own raw memory. Instead, it references a Storage object that manages memory lifetime.
     */

    class Tensor{
        
        public:

            /**
             * @brief Construct a tensor, empty or with specified shape and data type
             * @param shape Tensor shape
             * @param dtype Data Type
             * @param device Device
             */
            Tensor();
            Tensor(const Shape& shape, DataType dtype, Device device = Device(DeviceType::CPU));

            // ==================================
            // MetaData
            // ==================================
            [[nodiscard]] const Shape& shape()      const noexcept;
            [[nodiscard]] DataType dtype()          const noexcept;
            [[nodiscard]] size_t numel()            const noexcept;
            [[nodiscard]] size_t bytes()            const noexcept;
            [[nodiscard]] Device device()           const noexcept;
            [[nodiscard]] bool empty()              const noexcept;
            [[nodiscard]] size_t rank()             const noexcept;
            [[nodiscard]] size_t dim(size_t index)  const noexcept;
            [[nodiscard]] size_t offset()           const noexcept;
            [[nodiscard]] const Stride& stride()    const noexcept;

            [[nodiscard]] const std::shared_ptr<Storage>& storage() noexcept;

            [[nodiscard]] size_t element_offset(const size_t* indices) const noexcept;

            // ==================================
            // Property
            // ==================================
            [[nodiscard]] bool is_contiguous()       const noexcept;
            [[nodiscard]] bool has_storage()        const noexcept;

            /**
             * @brief Return a typed pointer to the underlying storage.
             *
             * The caller is responsible for ensuring T matches the tensor's runtime DataType.
             * ONLY FOR CONTIGUOUS TENSOR!!!!
             */
            template<typename T> [[nodiscard]] T* data(){
                assert(dtype_ == DataTypeTraits<T>::value);
                assert(storage_ != nullptr);
                assert(is_contiguous());
                return static_cast<T*>(storage_->data()) + offset_;
            }

            /**
             * @brief Return a const typed pointer to the underlying storage.
             * ONLY FOR CONTIGUOUS TENSOR!!!!
             */
            template<typename T> [[nodiscard]] const T* data() const{
                assert(dtype_ == DataTypeTraits<T>::value);
                assert(storage_ != nullptr);
                assert(is_contiguous());
                return static_cast<const T*>(storage_->data()) + offset_;
            }

            template<typename T> [[nodiscard]] T* raw_data(){
                assert(dtype_ == DataTypeTraits<T>::value);
                assert(storage_ != nullptr);

                return static_cast<T*>(storage_->data());
            }

            template<typename T> [[nodiscard]] const T* raw_data() const{
                assert(dtype_ == DataTypeTraits<T>::value);
                assert(storage_ != nullptr);

                return static_cast<T*>(storage_->data());
            }

            template<typename T, typename... Index> requires (std::is_integral_v<Index> && ...)
            T& at(Index... index){
                static_assert(sizeof...(Index) > 0, "Tensor index cannot be empty.");
                assert(sizeof...(Index) == shape_.rank());
                assert(dtype_ == DataTypeTraits<T>::value);
                size_t indices[] = {static_cast<size_t>(index)...};
                return raw_data<T>()[element_offset(indices)];
            } 

            template<typename T, typename... Index>  requires (std::is_integral_v<Index> && ...)
            const T& at(Index... index) const{
                static_assert(sizeof...(Index) > 0, "Tensor index cannot be empty.");
                assert(sizeof...(Index) == shape_.rank());
                assert(dtype_ == DataTypeTraits<T>::value);
                size_t indices[] = {static_cast<size_t>(index)...};
                return raw_data<T>()[element_offset(indices)];
            }



            static Tensor Create_View(const Tensor& base, Shape shape, Stride stride, size_t offset);
            [[nodiscard]] Tensor transpose(size_t dim0, size_t dim1) const;

        private:
            
            Tensor(
                const Shape& shape,
                DataType dtype,
                const std::shared_ptr<Storage>& storage,
                size_t offset,
                Stride stride,
                Device device = Device(DeviceType::CPU)
            );

            Shape shape_;
            DataType dtype_;
            std::shared_ptr<Storage> storage_;
            size_t offset_;
            Stride stride_;
            Device device_;



    };

}