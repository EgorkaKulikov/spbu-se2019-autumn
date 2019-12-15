
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#include "utils.hxx"

#define CUDA_assert(expr) do {                                                    \
	cudaError_t code = expr;                                                      \
	if (code != cudaSuccess) {                                                    \
		std::cerr << "Line: " << __LINE__ << " | " << #expr << std::endl;         \
		std::cerr << "    CUDA error: " << cudaGetErrorString(code) << std::endl; \
		return false;                                                             \
	}                                                                             \
} while (0)

#define MAX_THREADS 512
#define MAX_BLOCKS 32768

static __global__ void general_swap(int* data, int block_size, int pass, int div) {
	bitonic_swap(data, blockIdx.y * MAX_THREADS + threadIdx.x, blockIdx.x * div + threadIdx.y, block_size, pass);
}

static __global__ void remain_of_general_swap(int* data, int block_size, int pass, int offset) {
	bitonic_swap(data, blockIdx.y * MAX_THREADS + threadIdx.x, offset + threadIdx.y, block_size, pass);
}

static __global__ void remain_swap(int* data, int block_size, int pass, int number_of_blocks) {
	bitonic_swap(data, blockIdx.x * MAX_THREADS + threadIdx.x, number_of_blocks, block_size, pass);
}

static __global__ void remain_of_remain_swap(int* data, int block_size, int pass, int number_of_blocks, int offset) {
	bitonic_swap(data, offset + threadIdx.x, number_of_blocks, block_size, pass);
}

bool gpu_bitonic_sort(std::vector<int>& data) {
	int* device_data;

	CUDA_assert(cudaMalloc(&device_data, data.size() * sizeof(int)));
	CUDA_assert(cudaMemcpy(device_data, data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice));

	int number_of_stages = get_number_of_stages(data);

	for (int stage = 1; stage <= number_of_stages; stage++) {
		int block_size = 1 << stage;
		int number_of_blocks = data.size() >> stage;
		int remain_size = data.size() & (block_size - 1);

		for (int pass = 0; pass < stage; pass++) {
			int step = block_size >> 1;

			dim3 number_of_cuda_threads(min(MAX_THREADS, step));
			number_of_cuda_threads.y = MAX_THREADS / number_of_cuda_threads.x;
			
			dim3 number_of_cuda_blocks(number_of_blocks / number_of_cuda_threads.y, max(1, step / MAX_THREADS));

			if (number_of_cuda_blocks.x > 0) {
				general_swap<<<number_of_cuda_blocks, number_of_cuda_threads>>>(device_data, block_size, pass, number_of_cuda_threads.y);
				CUDA_assert(cudaGetLastError());
			}

			int block_offset = number_of_cuda_blocks.x * number_of_cuda_threads.y;
			number_of_cuda_blocks.x = number_of_blocks % number_of_cuda_threads.y;
			number_of_cuda_threads.y = 1;

			if (number_of_cuda_blocks.x > 0) {
				remain_of_general_swap<<<number_of_cuda_blocks, number_of_cuda_threads>>>(device_data, block_size, pass, block_offset);
				CUDA_assert(cudaGetLastError());
			}

			if (remain_size != 0) {
				int threads = remain_size - step;
				if (threads > 0) {
					int count = threads / MAX_THREADS;
					if (count > 0) {
						remain_swap<<<count, MAX_THREADS>>>(device_data, block_size, pass, number_of_blocks);
						CUDA_assert(cudaGetLastError());
					}

					int remain = threads % MAX_THREADS;
					if (remain > 0) {
						remain_of_remain_swap<<<1, remain>>>(device_data, block_size, pass, number_of_blocks, count * MAX_THREADS);
						CUDA_assert(cudaGetLastError());
					}
				}
			}

			CUDA_assert(cudaDeviceSynchronize());

			block_size >>= 1;
			number_of_blocks <<= 1;
		}
	}

	CUDA_assert(cudaMemcpy(data.data(), device_data, data.size() * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_assert(cudaFree(device_data));

	return true;
}
