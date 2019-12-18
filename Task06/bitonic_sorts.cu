#include "stdio.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define THREADS 1024

#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void swap(int* array, int first, int second)
{   
    int tmp = array[first];
    array[first] = array[second];
    array[second] = tmp;
}

__device__
void swap_gpu(int* array, int first, int second)
{   
    int tmp = array[first];
    array[first] = array[second];
    array[second] = tmp;
}

__global__
void bitonic_exchange_gpu(int* dev_values, int depth, unsigned long step)
{
    /* Sorting partners: i and pair_for_i */
    unsigned int i, pair_for_i; 
    
    /* Orient tells in which part of bitonic (sub-)sequence elements are (descending or ascending) */
    unsigned int orient_i, orient_pair_for_i; 

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pair_for_i = i + depth;

    orient_pair_for_i = pair_for_i & step;
    orient_i = i & step;

    /* 
        If current array[i] is the second for other array[j] (i<j) so we just do nothing
        It can be seen if for some a[i], a[pair_for_i] located in other bitonic (sub-)sequence
        For example, a[i] in ascending part, a[pair_for_i] in descending
    */
    if (orient_i != 0 && orient_pair_for_i == 0 
        || orient_i == 0 && orient_pair_for_i != 0)
    {
        return;
    }

    if (orient_i == 0) 
    {
        /* Sort ascending */
        if (dev_values[i]>dev_values[pair_for_i])
        {
            swap_gpu(dev_values, i, pair_for_i);
        }
	}
    else 
    {
		/* Sort descending */
        if (dev_values[i]<dev_values[pair_for_i])
        {
        	swap_gpu(dev_values, i, pair_for_i);
        }
	}
}

void bitonic_sort_gpu(int* array, unsigned long size)
{
    size_t size_mem_array = size * sizeof(int);
    int* array_gpu;

    gpu_error_check(cudaMalloc(&array_gpu, size_mem_array));
    gpu_error_check(cudaMemcpy(array_gpu, array, size_mem_array, cudaMemcpyHostToDevice));

    dim3 blocks = (size < THREADS) ? size : size / THREADS;
    dim3 threadsPerBlock = (size < THREADS) ? 1 : THREADS;

    for (int step = 2; step <= size; step <<= 1)
    {
        for (int depth = step >> 1; depth >= 1; depth >>= 1)
        {
            bitonic_exchange_gpu<<<blocks, threadsPerBlock>>>(array_gpu, depth , step);
        }
	}
	
	gpu_error_check(cudaMemcpy(array, array_gpu, size_mem_array, cudaMemcpyDeviceToHost));
	cudaFree(&array_gpu);
}


void bitonic_exchange(int* array, int depth, int step, unsigned long size)
{
    for (int i = 0; i < size; i++)
    {
        unsigned int pair_for_i;
        unsigned int orient_i, orient_pair_for_i;

        pair_for_i = i + depth;

        orient_pair_for_i = pair_for_i & step;
        orient_i = i & step;

        if (orient_i != 0 && orient_pair_for_i == 0 
            || orient_i == 0 && orient_pair_for_i != 0)
        {
            continue;
        }
        
        if (orient_i == 0)
        {
            if (array[i] > array[pair_for_i])
            {
                swap(array, i, pair_for_i);
            }
        } 
        else 
        {
            if (array[i] < array[pair_for_i])
            {
                swap(array, i, pair_for_i);
            }
        }
    }
}

void bitonic_sort(int* array, unsigned long size)
{   
    for (int step = 2; step <= size; step <<= 1)
    {
        for (int j = step >> 1 ; j >= 1; j >>= 1)
        {
            bitonic_exchange(array, j, step, size);
        }
    }
}