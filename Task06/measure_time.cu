#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "bitonic_sort.cuh"
#include "bitonic_sort_gpu.cuh"

//Binary logarithm of array length for bitonic sort
#define MAX_ARRAY_LOG_LEN 25

double get_exec_time(time_t start, time_t stop) {
	return (double)(stop - start) / CLOCKS_PER_SEC;
}

void fill_arr(int *arr, int arr_len) {
	srand(time(NULL));
	for (int i = 0; i < arr_len; i++) {
		arr[i] = rand();
	}
}

int main()
{
	FILE *output = fopen("measurements_output.txt", "w+");
	for (int log_len = 0; log_len < MAX_ARRAY_LOG_LEN; log_len++) {
		int arr_len = 1 << log_len;
		size_t arr_size = arr_len * sizeof(int);
		int *arr = (int *)malloc(arr_size);
		int *temp_arr = (int *)malloc(arr_size);
		fill_arr(arr, arr_len);
		clock_t start;

		//Initializing CUDA context
		cudaFree(0);

		//CPU version measurement
		memcpy(temp_arr, arr, arr_size);
		start = clock();
		bitonic_sort(temp_arr, log_len);
		double cpu_exec_time = get_exec_time(start, clock());

		//GPU version measurement
		memcpy(temp_arr, arr, arr_size);
		start = clock();
		bitonic_sort_gpu(temp_arr, log_len);
		double gpu_exec_time = get_exec_time(start, clock());

		//Measurements output for gnuplot
		fprintf(output, "%d %f %f\n", arr_len, gpu_exec_time, cpu_exec_time);
		free(arr);
		free(temp_arr);
	}
	fclose(output);

	//Correct exiting
	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}
