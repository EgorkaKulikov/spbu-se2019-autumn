
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <cstdio>
#include <random>
#include <time.h>
#include <iostream>

using namespace std;

const int CUDA_BLOCK_SIZE = 32;

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

__global__ void bitonicSwap(int* arr, int sz, int blockSize, int pass, int shift) {
	int block = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.z << shift) + blockIdx.y;
	int i = y * blockDim.y + threadIdx.y;
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
	if (cudaMalloc(&cudarr, sz * sizeof(int)) != cudaSuccess) {
		cerr << "Error when allocating device memory" << endl;
		exit(7);
	}

	if (cudaMemcpy(cudarr, arr, sz * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cerr << "Error when copying memory" << endl;
		exit(8);
	}

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
			int dimZ = step / threadsPerBlock.y;
			int dimY = 1;
			int shift = 0;
			while (dimY < dimZ) {
				dimY <<= 1;
				dimZ >>= 1;
				shift++;
			}

			dim3 numCUDABlocks(numBlocks / threadsPerBlock.x, dimY, dimZ);
			bitonicSwap <<< numCUDABlocks, threadsPerBlock >>> (cudarr, sz, blockSize, pass, shift);
			cudaError_t errSync = cudaGetLastError();
			cudaError_t errAsync = cudaDeviceSynchronize();
			if (errSync != cudaSuccess && errAsync != cudaSuccess) {
				cerr << "CUDA execution error" << endl;
				exit(9);
			}

			blockSize >>= 1;
			numBlocks <<= 1;
		}
	}

	if (cudaMemcpy(arr, cudarr, sz * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
		cerr << "Error when copying memory" << endl;
		exit(8);
	}

	cudaFree(cudarr);
}

int main(int argc, char **argv) {
	if (argc != 3) {
		cerr << "Invalid number of arguments!" << endl;
		return 1;
	}

	FILE* input = fopen(argv[1], "r");
	if (input == nullptr) {
		cerr << "Error while opening file!" << endl;
		return 2;
	}

	int sz;
	if (fscanf(input, "%d", &sz) != 1) {
		cerr << "Error while reading file!" << endl;
		return 3;
	}

	if (sz == 0) {
		return 0;
	}

	int padded_sz = 1;
	while (padded_sz < sz) {
		padded_sz <<= 1;
	}

	int* nums = new int[padded_sz];
	if (nums == nullptr) {
		cerr << "Error when allocating host memory!" << endl;
		return 4;
	}

	int max_value = (1 << 31);
	for (int i = 0; i < sz; i++) {
		if (fscanf(input, "%d", nums + i) != 1) {
			cerr << "Error while reading file!" << endl;
			return 3;
		}
		max_value = max(max_value, nums[i]);
	}

	fclose(input);
	for (int i = sz; i < padded_sz; i++) {
		nums[i] = max_value;
	}
	
	int time_count = -1;
	if (argv[2][0] == 'l') {
		int timestamp = clock();
		bitonicSort(nums, padded_sz);
		time_count = clock() - timestamp;
	}
	else if (argv[2][0] == 'p') {
		int timestamp = clock();
		bitonicSortCUDA(nums, padded_sz);
		time_count = clock() - timestamp;
	}
	else {
		cerr << "Invalid sorting mode parameter!" << endl;
		return 5;
	}

	int prev = nums[0];
	for (int i = 1; i < padded_sz; i++) {
		if (nums[i] < prev) {
			cerr << "Invalid sorted order!" << endl;
			for (int j = i - 5; j <= i + 5; j++) {
				cerr << nums[j] << " ";
			}
			cerr << endl;
			cerr << i << " " << sz << " " << padded_sz << " " << prev << " " << nums[i] << endl;
			//return 6;
		}
		prev = nums[i];
	}
	
	cout << "The sort was completed in " << time_count << "ms, " << sz << " elements sorted" << endl;
	delete[] nums;
	return 0;
}
