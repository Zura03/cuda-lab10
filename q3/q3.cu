#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void oddEvenSort(int* array, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int phase = 0; phase < n; phase++) {
		if (phase % 2 == 0) {
			if ((tid % 2 == 0) && (tid < n - 1)) {
				if (array[tid] > array[tid + 1]) {
					int temp = array[tid];
					array[tid] = array[tid + 1];
					array[tid + 1] = temp;
				}
			}
		}
		else {
			if ((tid % 2 != 0) && (tid < n - 1)) {
				if (array[tid] > array[tid + 1]) {
					int temp = array[tid];
					array[tid] = array[tid + 1];
					array[tid + 1] = temp;
				}
			}
		}
		__syncthreads();
	}
}

int main() {
	int* array, n;
	int* d_arr;

	printf("Enter n: ");
	scanf("%d", &n);

	int size = n * sizeof(int);
	array = (int*)malloc(size);

	printf("Enter the array: ");
	for (int i = 0; i < n; i++)
		scanf("%d", &array[i]);

	cudaMalloc((void**)&d_arr, size);
	cudaMemcpy(d_arr, array, size, cudaMemcpyHostToDevice);

	dim3 gridSize(1, 1);
	dim3 blockSize(n, 1);

	oddEvenSort << <gridSize, blockSize >> > (d_arr, n);

	cudaMemcpy(array, d_arr, size, cudaMemcpyDeviceToHost);

	printf("Sorted array: ");
	for (int i = 0; i < n; i++)
		printf("%d\t", array[i]);

	cudaFree(d_arr);
	return 0;
}