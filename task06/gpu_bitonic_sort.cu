
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#include "utils.hxx"

#define CUDA_assert(expr) do { \
	cudaError_t code = expr; \
	if (code != cudaSuccess) { \
		std::cerr << "Line: " << __LINE__ << " | " << #expr << std::endl; \
		std::cerr << "    CUDA error: " << cudaGetErrorString(code) << std::endl; \
		return false; \
	} \
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

struct cuda_config {
	dim3 tnum;
	dim3 bnum;
};

#define run_on_device(func, config, ...) do { \
	if (config.tnum.x == 0 || config.tnum.y == 0 || config.bnum.x == 0 || config.bnum.y == 0) { \
		break; \
	} \
	func<<<config.bnum, config.tnum>>>(__VA_ARGS__); \
	CUDA_assert(cudaGetLastError()); \
} while (0)

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

			struct cuda_config config;
			config.tnum.x = min(MAX_THREADS, step);
			config.tnum.y = MAX_THREADS / config.tnum.x;
			config.bnum.x = number_of_blocks / config.tnum.y;
			config.bnum.y = max(1, step / MAX_THREADS);

			run_on_device(general_swap, config, device_data, block_size, pass, config.tnum.y);

			int block_offset = config.bnum.x * config.tnum.y;
			config.bnum.x = number_of_blocks % config.tnum.y;
			config.tnum.y = 1;

			run_on_device(remain_of_general_swap, config, device_data, block_size, pass, block_offset);

			if (remain_size != 0) {
				int threads = remain_size - step;
				if (threads > 0) {
					config.bnum = threads / MAX_THREADS;
					config.tnum = MAX_THREADS;

					run_on_device(remain_swap, config, device_data, block_size, pass, number_of_blocks);

					int offset = config.bnum.x * MAX_THREADS;
					config.bnum = 1;
					config.tnum = threads % MAX_THREADS;

					run_on_device(remain_of_remain_swap, config, device_data, block_size, pass, number_of_blocks, offset);
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
