#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void a(int i) {
    printf("hello");
    i = 8;
}

__global__ void kernel(int* i) {
    printf("in kernel");
    *i = 3;
    a(*i);
}

int main() {
    int* d_i;  // Pointer for GPU memory

    // Allocate memory on GPU
    cudaMalloc((void**)&d_i, sizeof(int));

    // Initialize host variable
    int h_i = 5;

    // Copy host variable to GPU
    cudaMemcpy(d_i, &h_i, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    kernel << <1, 1 >> > (d_i);

    // Copy the result back to host
    cudaMemcpy(&h_i, d_i, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_i);

    return 0;
}
