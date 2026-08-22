#include <cstdio>

__global__ void hello_kernel(){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU! global_id=%d\n", global_id);
}

int main(){
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}