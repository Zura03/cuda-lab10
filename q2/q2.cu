/*
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

const int N = 8;

__global__ void merge(int* arr, int* temp, int left, int middle, int right) {
	int i = left;
	int j = middle;
	int k = right;

	while (i < middle && j < right) {
		if (arr[i] <= arr[j]) {
			temp[k++] = arr[i++];
		}
		else {
			temp[k++] = arr[j++];
		}
	}

	while (j < middle) {
		temp[k++] = arr[i++];
	}

	while (j < right) {
		temp[k++] = arr[j++];
	}
}
__global__ void mergesort(int* arr, int* temp, int left, int right) {
	if (right - left <= 1)
		return;

	int middle = (left + right) / 2;

	mergesort << <1, 1 >> > (arr, temp, left, middle);
	mergesort << <1, 1 >> > (arr, temp, middle, right);

	cudaDeviceSynchronize();

	merge << <1, 1 >> > (arr, temp, left, middle, right);

	cudaDeviceSynchronize();
}
int main() {
	int host_arr[N] = { 10, 27, 42,3, 9, 82, 11, 65 };

	int* d_arr, * d_temp;

	cudaMalloc((void**)&d_arr, N * sizeof(int));
	cudaMalloc((void**)d_temp, N * sizeof(int));

	cudaMemcpy(d_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_temp, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);

	mergesort << <1, 1 >> > (d_arr, d_temp, 0, N);

	cudaMemcpy(host_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("sorted array: ");
	for (int i = 0; i < N; i++)
		printf("%d\t", host_arr[i]);

	cudaFree(d_arr);
	cudaFree(d_temp);
	return 0;
}*/

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i=0, j=0, k=0;

    while(i<m && j<n) 
        if(A[i] <= B[j]) 
            C[k++] = A[i++];
        else 
            C[k++] = B[j++];
    
    if(i == m) 
        while(j < n) 
            C[k++] = B[j++];
    else 
        while(i < m) 
            C[k++] = A[i++];
}

__device__ int co_rank(int k, int* A, int m, int* B, int n) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k-n;
    int j_low = 0 > (k-m) ? 0 : k-m;
    int delta;
    int active = 1;

    while(active) {
        if(i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if(j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else {
            active = 0;
        }
    }

    return i;
}

__global__ void merge_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int elementsPerThread = ceil((double)(m+n)/(blockDim.x*gridDim.x));
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m+n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}

int main() {
    int *A, *B, *C;
    int m, n, sizeA, sizeB, sizeC;
    int *d_A, *d_B, *d_C;

    printf("Enter the size of the first array, A: ");
    scanf("%d", &m);

    printf("Enter the size of the second array, B: ");
    scanf("%d", &n);

    sizeA = sizeof(int) * m;
    sizeB = sizeof(int) * n;
    sizeC = sizeof(int) * (m+n);

    A = (int*) malloc(sizeA);
    B = (int*) malloc(sizeB);
    C = (int*) malloc(sizeC);

    printf("Enter the sorted array, A: ");
    for(int i=0; i<m; i++) 
        scanf("%d", A+i);

    printf("Enter the sorted array, B: ");
    for(int i=0; i<n; i++) 
        scanf("%d", B+i);

    cudaMalloc((void**) &d_A, sizeA);
    cudaMalloc((void**) &d_B, sizeB);
    cudaMalloc((void**) &d_C, sizeC);
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    merge_kernel<<<1, ceil((m+n)/5.0)>>>(d_A, m, d_B, n, d_C);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("Resultant Array after Parallel Merge Sorting:\n");
    for(int i=0; i<m+n; i++) 
        printf("%4d", C[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}