#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float* A, const float* B, float* C, size_t N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;      // ???
    if(i < N){                                          // ???
        C[i] = A[i] + B[i];
    }
}

int main(){
    constexpr size_t N = 1024;
    size_t bytes = N * sizeof(float);

    // Host Memory, ???
    float* h_A = static_cast<float*>(malloc(bytes));
    float* h_B = static_cast<float*>(malloc(bytes));
    float* h_C = static_cast<float*>(malloc(bytes));

    for(size_t i = 0; i < N; ++i){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device Memory, ???
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Host -> Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    constexpr int threads_per_block = 256;
    int block = (N + threads_per_block - 1) / threads_per_block;

    vector_add<<<block, threads_per_block>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Device -> Host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Check result
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            printf("ERROR: i=%d expected=%f got=%f\n", i, expected, h_C[i]);
            return 1;
        }
    }

    printf("VectorAdd PASS\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}