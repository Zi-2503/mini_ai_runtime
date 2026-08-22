#include "runtime/memory/storage.h"

#include <cstdlib>

namespace runtime{

    Storage::Storage(size_t bytes, const Allocator& allocator)
        : data_(allocator.Allocate(bytes)),
          bytes_(bytes),
          allocator_(allocator)
        {}

    Storage::~Storage(){
        allocator_.Free(data_);
    }

    void* Storage::data() noexcept{
        return data_;
    }

    const void* Storage::data() const noexcept{
        return data_;
    }

    size_t Storage::bytes() noexcept{
        return bytes_;
    }

}