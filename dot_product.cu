#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dot_product.h"
#include <iostream>
#include <vector>


using namespace std;

#define TILE_SIZE 16

__global__ void mat_mul_gpu(float* matA, float* matB, float* matResult, int matARows, int matACols, int matBCols) {

	int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
	int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

	int linearOffset = globalRow * matBCols + globalCol;

	if ((globalRow < matARows) && (globalCol < matBCols)) {
		float tempSum = 0.0;
		for (int idx = 0; idx < matACols; idx++) {
			tempSum += matA[globalRow * matACols + idx] * matB[idx * matBCols + globalCol];
		}
		matResult[linearOffset] = tempSum;
	}
}

void dot_product_cu(dot_product_cu_parameters params) {
	size_t sizeA = params.vec_one_row * params.vec_one_col * sizeof(float);
	size_t sizeB = params.vec_two_row * params.vec_two_col * sizeof(float);
	size_t sizeResult = params.vec_one_row * params.vec_two_col * sizeof(float);

	float* d_matA;
	float* d_matB;
	float* d_matResult;
	cudaMalloc((void**)&d_matA, sizeA);
	cudaMalloc((void**)&d_matB, sizeB);
	cudaMalloc((void**)&d_matResult, sizeResult);

	cudaMemcpy(d_matA, params.pbuf1_ptr, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matB, params.pbuf2_ptr, sizeB, cudaMemcpyHostToDevice);

	unsigned int gridRows = (params.vec_one_row + TILE_SIZE - 1) / TILE_SIZE;
	unsigned int gridCols = (params.vec_two_col + TILE_SIZE - 1) / TILE_SIZE;
	dim3 gridDimensions(gridCols, gridRows);
	dim3 blockDimensions(TILE_SIZE, TILE_SIZE);
	mat_mul_gpu<<<gridDimensions, blockDimensions>>>(d_matA, d_matB, d_matResult, params.vec_one_row, params.vec_one_col, params.vec_two_col);

	cudaDeviceSynchronize();

	cudaMemcpy(params.pbuf_ret_ptr, d_matResult, sizeResult, cudaMemcpyDeviceToHost);
}
