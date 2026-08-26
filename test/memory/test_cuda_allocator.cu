#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#include "runtime/memory/allocator/cuda_allocator.h"

using namespace runtime;

void TestCUDAAllocator() {

    CUDAAllocator allocator;

    void* ptr = allocator.Allocate(1024);

    assert(ptr != nullptr);

    allocator.Free(ptr);

    std::cout << "[PASS] CUDAAllocator\n";
}

int main() {

    TestCUDAAllocator();

    return 0;
}