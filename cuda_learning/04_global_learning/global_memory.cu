#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::fprintf(                                     \
                stderr,                                       \
                "CUDA error at %s:%d: %s\n",                  \
                __FILE__,                                     \
                __LINE__,                                     \
                cudaGetErrorString(err));                     \
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
// Non-contiguous
//
// Logical tensor:
//     shape  = [N, N]
//     stride = [1, N]
//
// Physical storage is still N*N elements.
//
// logical A[row][col]
//     -> A[row * 1 + col * N]
//
// ============================================================

__global__ void copy_noncontiguous(const float* A, float* B, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        int offset = row + col * N;
        B[idx] = A[offset];
    }
}


int main(){
    constexpr int N = 1024;
    constexpr int total = N * N;
    const size_t bytes = total * sizeof(float);

    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    for (int i = 0; i < total; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Launch configuration
    // --------------------------------------------------------

    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    // --------------------------------------------------------
    // Contiguous
    // --------------------------------------------------------

    copy_contiguous<<<blocks, threads>>>(d_A, d_B, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; ++i) {
        if (h_B[i] != h_A[i]) {
            std::printf("Contiguous FAILED at %d\n", i);
            return 1;
        }
    }

    std::printf("Contiguous: PASS\n");

    // --------------------------------------------------------
    // Non-contiguous
    // --------------------------------------------------------

    copy_noncontiguous<<<blocks, threads>>>(d_A, d_B, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    // We are deliberately reading the physical matrix
    // with transpose-like strides, so B will not equal A.
    std::printf("Non-contiguous kernel: PASS (executed)\n");

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    std::free(h_A);
    std::free(h_B);

    return 0;
}