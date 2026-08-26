#include <cassert>
#include <iostream>

#include "runtime/memory/allocator/cpu_allocator.h"

using namespace runtime;

void TestCPUAllocator();

int main() {
    TestCPUAllocator();
}

void TestCPUAllocator() {
    CPUAllocator allocator;
    void* ptr = allocator.Allocate(1024);
    assert(ptr != nullptr);
    allocator.Free(ptr);
    std::cout << "[PASS] CPUAllocator\n";
}