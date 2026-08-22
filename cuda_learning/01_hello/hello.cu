#include <cstdio>

__global__ void hello_kernel(){
    printf("Hello from GPU! block = %d, thread = %d", blockIdx.x, threadIdx.x);
}

int main(){
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}