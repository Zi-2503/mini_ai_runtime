#include <benchmark/benchmark.h>

#include <cblas.h>

#include "runtime/core/device.h"
#include "runtime/core/dtype.h"
#include "runtime/tensor/tensor.h"

using namespace runtime;


static void BM_OpenBLAS_SGEMM(benchmark::State& state)
{
    const int M = state.range(0);
    const int N = state.range(1);
    const int K = state.range(2);

    Tensor A(Shape({static_cast<size_t>(M), static_cast<size_t>(K)}), DataType::Float32, Device(DeviceType::CPU));
    Tensor B(Shape({static_cast<size_t>(K), static_cast<size_t>(N)}), DataType::Float32, Device(DeviceType::CPU));
    Tensor C(Shape({static_cast<size_t>(M), static_cast<size_t>(N)}), DataType::Float32, Device(DeviceType::CPU));

    float* a = A.data<float>();
    float* b = B.data<float>();
    float* c = C.data<float>();

    for (size_t i = 0; i < A.numel(); ++i) {
        a[i] = 1.0f;
    }

    for (size_t i = 0; i < B.numel(); ++i) {
        b[i] = 1.0f;
    }

    for (auto _ : state) {

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,

            1.0f,
            a, K,
            b, N,

            0.0f,
            c, N
        );
    }

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    state.counters["GFLOPS"] = benchmark::Counter(flops / 1e9, benchmark::Counter::kIsIterationInvariantRate);
    
}


BENCHMARK(BM_OpenBLAS_SGEMM)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024});


BENCHMARK_MAIN();