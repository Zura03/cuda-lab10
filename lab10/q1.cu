#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#define MAX_MASK_WIDTH 5
__constant__ float M[MAX_MASK_WIDTH];

__global__ void convolution1D(float* N, float* P, int MaskWidth, int Width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;
	int N_start_point = i - (MaskWidth / 2);
	for (int j = 0; j < MaskWidth; j++) {
		if (N_start_point + j > 0 && N_start_point + j < Width)
			Pvalue += N[N_start_point + j] * M[j];
	}
	P[i] = Pvalue;
}

int main() {
	float* N, * P, * h_m;
	int MaskWidth, Width;
	float* d_N, * d_P;
	
	printf("Enter size of vector: ");
	scanf("%d", &Width);
	
	N = (float*)malloc(Width * sizeof(float));
	P = (float*)malloc(Width * sizeof(float));

	printf("Enter vector: ");
	for (int i = 0; i < Width; i++)
		scanf("%f", &N[i]);

	printf("Enter size of mask: ");
	scanf("%d", &MaskWidth);
	h_m = (float*)malloc(MaskWidth * sizeof(float));
	printf("Enter mask: ");
	for (int i = 0; i < MaskWidth; i++)
		scanf("%f", &h_m[i]);

	cudaMalloc((void**)&d_N, Width * sizeof(float));
	cudaMalloc((void**)&d_P, Width * sizeof(float));

	cudaMemcpyToSymbol(M, h_m, MaskWidth * sizeof(float));
	cudaMemcpy(d_N, N, Width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_P, P, Width * sizeof(float), cudaMemcpyHostToDevice);

	int blocksize = 32;
	dim3 gridSize((Width + blocksize - 1) / blocksize, 1, 1);
	dim3 blockSize(blocksize, 1, 1);

	convolution1D << <gridSize, blockSize >> > (d_N, d_P, MaskWidth, Width);

	cudaMemcpy(P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

	printf("convolved vector: ");
	for (int i = 0; i < Width; i++)
		printf("%f\t", P[i]);

	cudaFree(d_N);
	cudaFree(d_P);
	return 0;
}