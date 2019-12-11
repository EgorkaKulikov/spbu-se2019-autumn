
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <cstdio>
#include <random>
#include <time.h>
#include <iostream>

using namespace std;

const int CUDA_BLOCK_SIZE = 16;

void bitonicSort(int* arr, int sz) {
	int numStages = 0;
	for (int i = sz; i > 0; i >>= 1, numStages++);
	numStages--;
	for (int stage = 1; stage <= numStages; stage++) {
		int numPasses = stage;
		int blockSize = 1 << stage;
		int numBlocks = sz >> stage;
		for (int pass = 0; pass < numPasses; pass++) {
			int step = blockSize >> 1;
			for (int block = 0; block < numBlocks; block++) {
				for (int i = 0; i < step; i++) {
					int index = block * blockSize + i;
					bool ascending = ((block >> pass) & 1) == 0;
					if ((ascending && arr[index] > arr[index + step])
							|| (!ascending && arr[index] < arr[index + step])) {
						int tmp = arr[index];
						arr[index] = arr[index + step];
						arr[index + step] = tmp;
					}
				}
			}

			blockSize >>= 1;
			numBlocks <<= 1;
		}
	}
}

__global__ void bitonicSwap(int* arr, int sz, int blockSize, int pass) {
	int block = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int step = blockSize >> 1;
	int index = block * blockSize + i;
	bool ascending = ((block >> pass) & 1) == 0;
	if ((ascending && arr[index] > arr[index + step])
		|| (!ascending && arr[index] < arr[index + step])) {
		int tmp = arr[index];
		arr[index] = arr[index + step];
		arr[index + step] = tmp;
	}
}

void bitonicSortCUDA(int* arr, int sz) {
	int* cudarr;
	cudaMalloc(&cudarr, sz * sizeof(int));
	cudaMemcpy(cudarr, arr, sz * sizeof(int), cudaMemcpyHostToDevice);
	int numStages = 0;
	for (int i = sz; i > 0; i >>= 1, numStages++);
	numStages--;
	for (int stage = 1; stage <= numStages; stage++) {
		int numPasses = stage;
		int blockSize = 1 << stage;
		int numBlocks = sz >> stage;
		for (int pass = 0; pass < numPasses; pass++) {
			int step = blockSize >> 1;
			dim3 threadsPerBlock(min(CUDA_BLOCK_SIZE, numBlocks), min(step, CUDA_BLOCK_SIZE));
			dim3 numCUDABlocks(numBlocks / threadsPerBlock.x, step / threadsPerBlock.y);
			bitonicSwap <<< numCUDABlocks, threadsPerBlock >>> (cudarr, sz, blockSize, pass);
			blockSize >>= 1;
			numBlocks <<= 1;
		}
	}

	cudaMemcpy(arr, cudarr, sz * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cudarr);
}

int main() {
	srand(time(0));
	int sz = 1 << 24;
	int maxValue = 100000;
	int* nums = new int[sz];
	int* cuda_nums = new int[sz];
	for (int i = 0; i < sz; i++) {
		nums[i] = rand() % maxValue;
		cuda_nums[i] = nums[i];
	}

	int timestamp = clock();
	bitonicSort(nums, sz);
	cout << clock() - timestamp << endl;
	timestamp = clock();
	bitonicSortCUDA(cuda_nums, sz);
	cout << clock() - timestamp << endl;
	delete[] nums;
	delete[] cuda_nums;
	return 0;
}
