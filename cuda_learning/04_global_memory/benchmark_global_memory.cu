#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>


#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::fprintf(                                     \
                stderr,                                       \
                "CUDA error at %s:%d: %s\n",                 \
                __FILE__,                                     \
                __LINE__,                                     \
                cudaGetErrorString(err));                    \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// Contiguous
// ============================================================

__global__ void copy_contiguous(const float* A, float* B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        B[idx] = A[idx];
    }
}


// ============================================================
// Strided
// stride = [1, N]
// ============================================================

__global__ void copy_strided(const float* A, float* B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        int offset = row + col * N;
        B[idx] = A[offset];
    }
}


// ============================================================
// Benchmark helper
// ============================================================

float benchmark_kernel(const float* d_A, float* d_B, int N, bool strided, int iterations){
    constexpr int threads = 256;
    int total = N * N;
    int blocks = (total + threads - 1) / threads;

    // Warmup
    if (strided) {
        copy_strided<<<blocks, threads>>>(d_A, d_B, N);
    } else {
        copy_contiguous<<<blocks, threads>>>(d_A, d_B, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // Events
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {

        if (strided) {
            copy_strided<<<blocks, threads>>>(d_A, d_B, N);
        } else {
            copy_contiguous<<<blocks, threads>>>(d_A, d_B, N);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));


    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    return milliseconds / iterations;
}


// ============================================================
// Main
// ============================================================

int main(){
    constexpr int N = 4096;
    constexpr int iterations = 100;

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    // --------------------------------------------------------
    // Allocate device memory
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));


    // --------------------------------------------------------
    // Initialize
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemset(d_A, 0, bytes));

    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    float contiguous_ms = benchmark_kernel(d_A, d_B, N, false, iterations);
    float strided_ms = benchmark_kernel(d_A, d_B, N, true, iterations);
        

    // --------------------------------------------------------
    // Result
    // --------------------------------------------------------

    std::printf("N = %d\n", N);
    std::printf("Iterations = %d\n\n", iterations);
    std::printf("Contiguous : %.4f ms\n", contiguous_ms);
    std::printf("Strided    : %.4f ms\n", strided_ms);
    std::printf("Ratio      : %.2fx\n", strided_ms / contiguous_ms);

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return 0;
}