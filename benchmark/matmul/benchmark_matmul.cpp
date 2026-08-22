#include <benchmark/benchmark.h>

#include "runtime/core/device.h"
#include "runtime/core/dtype.h"
#include "runtime/execution/execution_context.h"
#include "runtime/kernel/cpu/cpu_matmul_kernel.h"
#include "runtime/tensor/tensor.h"

using namespace runtime;

static void BM_CPU_MatMul(benchmark::State& state)
{
    const size_t M = state.range(0);
    const size_t N = state.range(1);
    const size_t K = state.range(2);

    Tensor A(Shape({M, K}), DataType::Float32, Device(DeviceType::CPU));
    Tensor B(Shape({K, N}), DataType::Float32, Device(DeviceType::CPU));
    Tensor C(Shape({M, N}), DataType::Float32, Device(DeviceType::CPU));

    // ========================================
    // Initialize
    // ========================================

    float* a = A.data<float>();
    float* b = B.data<float>();

    for (size_t i = 0; i < A.numel(); ++i) {
        a[i] = 1.0f;
    }

    for (size_t i = 0; i < B.numel(); ++i) {
        b[i] = 1.0f;
    }

    ExecutionContext context;

    context.AddInput(A);
    context.AddInput(B);
    context.AddOutput(C);

    CPUMatMulKernel<float> kernel;

    // ========================================
    // Benchmark
    // ========================================

    for (auto _ : state) {

        Status status = kernel.Execute(context);

        if (!status.ok()) {
            state.SkipWithError(status.message().c_str());
        }
    }

    // ========================================
    // Performance
    // ========================================

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    state.counters["GFLOPS"] = benchmark::Counter(flops / 1e9, benchmark::Counter::kIsIterationInvariantRate);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_CPU_MatMul)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024});

BENCHMARK_MAIN();