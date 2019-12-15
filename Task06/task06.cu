#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

const int BLOCK_NUM = 1024;

#pragma region CPU
void bitonic_merge(int *a, int low, int cnt, int dir)
{
	if (cnt > 1)
	{
		int k = cnt / 2;
		for (int i = low; i < low + k; i++)
		{
			if (dir == (a[i] > a[i + k]))
			{
				std::swap(a[i], a[i + k]);
			}
		}
		bitonic_merge(a, low, k, dir);
		bitonic_merge(a, low + k, k, dir);
	}
}

void bitonic_sort_cpu(int *a, int low, int cnt, int dir)
{
	if (cnt > 1)
	{
		int k = cnt / 2;
		bitonic_sort_cpu(a, low, k, 1);
		bitonic_sort_cpu(a, low + k, k, 0);

		bitonic_merge(a, low, cnt, dir);
	}
}
#pragma endregion CPU

#pragma region GPU
#define GPUERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void bitonic_sort_step(int *a, int j, int k) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int ixj = tid ^ j;

	if (ixj > tid)
	{
		if ((tid & k) == 0)
		{
			if (a[tid] > a[ixj])
			{
				int temp;
				temp = a[tid];
				a[tid] = a[ixj];
				a[ixj] = temp;
			}
		}
		else
			if (a[tid] < a[ixj])
			{
				int temp;
				temp = a[tid];
				a[tid] = a[ixj];
				a[ixj] = temp;
			}
		}
}

void bitonic_sort_gpu(int *arr, int cnt) {
	int *a;
	GPUERRCHK(cudaMalloc((void**)&a, cnt * sizeof(int)));
	GPUERRCHK(cudaMemcpy(a, arr, cnt * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blocks = (cnt < BLOCK_NUM) ? 1 : BLOCK_NUM;;
	dim3 threads = (cnt < BLOCK_NUM) ? cnt : cnt / BLOCK_NUM;;

	for (unsigned int k = 2; k <= cnt; k *= 2)
	{
		for (unsigned int j = k / 2; j > 0; j /= 2)
		{
			bitonic_sort_step <<<blocks, threads >> > (a, j, k);
		}
	}
	GPUERRCHK(cudaMemcpy(arr, a, cnt * sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(a);

}
#pragma endregion GPU

int main()
{
	int cnt = 0;
	const int n = pow(2, 20);
	int *arr = (int*)malloc(1e9 * sizeof(int));
	for (cnt = 0; cnt < n; cnt++)
	{
		arr[cnt] = rand() % 10000;
	}

#pragma region cpu_time
	auto begin = std::chrono::steady_clock::now();
	bitonic_sort_cpu(arr, 0, cnt, 1);
	auto end = std::chrono::steady_clock::now();

	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << "The time: " << elapsed_ms.count() << " ms\n";
#pragma endregion cpu_time

#pragma region gpu_time
	auto begin = std::chrono::steady_clock::now();
	bitonic_sort_gpu(arr, cnt);
	auto end = std::chrono::steady_clock::now();

	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << "The time: " << elapsed_ms.count() << " ms\n";
#pragma endregion gpu_time
    return 0;
}
