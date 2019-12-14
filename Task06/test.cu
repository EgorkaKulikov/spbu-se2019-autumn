#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "bitonic_sort.cuh"
#include "bitonic_sort_gpu.cuh"

//Binary logarithm of array length for bitonic sort
#define MAX_ARRAY_LOG_LEN 20

bool is_sorted(int *arr, int len) {
	bool result = true;
	for (int i = 0; i < len - 1; i++) {
		result = arr[i] <= arr[i + 1];
	}
	return result;
}

void fill_arr(int *arr, int arrLen) {
	srand(time(NULL));
	for (int i = 0; i < arrLen; i++) {
		arr[i] = rand();
	}
}

int main()
{
	int arr_len = 1 << MAX_ARRAY_LOG_LEN;
	size_t arr_size = arr_len * sizeof(int);
	int *arr = (int *)malloc(arr_size);
	int *temp_arr = (int *)malloc(arr_size);

	fill_arr(arr, arr_len);

	//Initializing CUDA context
	cudaFree(0);

	//CPU version test
	memcpy(temp_arr, arr, arr_size);
	bitonic_sort(temp_arr, MAX_ARRAY_LOG_LEN);
	if (!is_sorted(temp_arr, arr_len)) {
		printf("FAIL FOR LENGTH %d\n", arr_len);
	}
	else {
		printf("CPU version SUCCESS\n");
	}

	//GPU version test
	memcpy(temp_arr, arr, arr_size);
	bitonic_sort_gpu(temp_arr, MAX_ARRAY_LOG_LEN);
	if (!is_sorted(temp_arr, arr_len)) {
		printf("FAIL FOR LENGTH %d\n", arr_len);
	}
	else {
		printf("GPU version SUCCESS\n");
	}

	//Correct exiting
	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}
