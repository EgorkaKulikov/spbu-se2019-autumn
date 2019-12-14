#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define NUM_THREADS 256
#define gpu_errcheck(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void bitonic_sort_step(int *arr, int comp_dist, int seq_size) {
	int item = threadIdx.x + blockIdx.x * blockDim.x;
	int pair_item = item | comp_dist;
	bool in_first_half = (item & seq_size) == 0;

	//First half of the sequence is sorted ascending, second - descending
	if (in_first_half && arr[item] > arr[pair_item]
		|| !in_first_half && arr[item] < arr[pair_item]) {
		int temp;
		temp = arr[item];
		arr[item] = arr[pair_item];
		arr[pair_item] = temp;
	}
}

void bitonic_sort_gpu(int *arr, unsigned int log_len) {
	int arr_len = 1 << log_len;
	size_t arr_size = arr_len * sizeof(int);
	int* gpu_arr;

	gpu_errcheck(cudaMalloc(&gpu_arr, arr_size));
	gpu_errcheck(cudaMemcpy(gpu_arr, arr, arr_size, cudaMemcpyHostToDevice));

	dim3 num_blocks = arr_len / NUM_THREADS;
	dim3 num_threads = NUM_THREADS;

	//Case for small array
	if (arr_len / NUM_THREADS == 0) {
		num_threads = 1;
		num_blocks = arr_len;
	}
	for (int seq_size = 2; seq_size <= arr_len; seq_size <<= 1) {
		//Comparison distance loop
		for (int comp_dist = seq_size >> 1; comp_dist > 0; comp_dist >>= 1) {
			bitonic_sort_step <<< num_blocks, num_threads >>> (gpu_arr, comp_dist, seq_size);
		}
	}

	gpu_errcheck(cudaMemcpy(arr, gpu_arr, arr_size, cudaMemcpyDeviceToHost));
	cudaFree(&gpu_arr);
}
