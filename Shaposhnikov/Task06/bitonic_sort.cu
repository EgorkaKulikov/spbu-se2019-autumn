#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define THREADS 512

#define GPUERRORCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
        exit(code);
   }
}

int next_power_of_two(int *array, int sz) {
  int log_size = 0;
  for (int i = sz; i > 0; i >>= 1)
    log_size++;
  
  log_size--;
  if (sz != (1 << log_size))
    return (log_size + 1);
  else
    return log_size;
}

//GPU version: 
__global__ void gpu_bitonic_sort_step(int *array, int phase_val, int step_val)
{
  unsigned num , p_num; //it's 2 numbers that are to be compared and swapped
  num = threadIdx.x + blockDim.x * blockIdx.x;
  bool is_in_first = (num & step_val) == 0;
  p_num = num ^ phase_val;

  if (p_num < num) {
    //so threads don't compare same numbers twice
    return;
  }

  if (is_in_first && array[num] > array[p_num]) {
    //ascending sequence
    int temp = array[num];
    array[num] = array[p_num];
    array[p_num] = temp;
  }
  if (!is_in_first && array[num] < array[p_num]) {
    //descending sequence
    int temp = array[num];
    array[num] = array[p_num];
    array[p_num] = temp;
  }
}

extern "C"
void gpu_bitonic_sort(int *arr, int sz)
{
  int *gpu_array;
  size_t size_old = sz * sizeof(int);
 
  int log_new_size = next_power_of_two(arr, sz);
  int size = 1 << log_new_size;
  printf("Size to be allocated is: %d\n", size);

  int diff = size - sz;

  int *temp = (int *) malloc(size * sizeof(int));
  memcpy(temp, arr, sz * sizeof(int));

  //fill empty cells with max values that then'll be not copied
  for (int i = 0; i < diff; i++) {
      temp[sz + i] = INT_MAX; 
  }

  GPUERRORCHECK(cudaMalloc(&gpu_array, size * sizeof(int)));
  GPUERRORCHECK(cudaMemcpy(gpu_array, temp, size * sizeof(int), cudaMemcpyHostToDevice));
  free(temp);

  //setting appropriate amounts of threads and blocks 
  dim3 blocks = (size < THREADS) ? size : size / THREADS;
  dim3 threads = (size < THREADS) ? 1 : THREADS;

  //it's like first for-loop stands for steps, 2nd - for phases
  for (int step_val = 2; step_val <= size; step_val <<= 1) {
    for (int phase_val = step_val >> 1; phase_val > 0; phase_val >>= 1) {
      gpu_bitonic_sort_step <<<blocks, threads>>> (gpu_array, phase_val, step_val);
    }
  }

  //and with size_old it's supposed that these allocated INT_MAXs are not touched
  //as it locates at the end of gpu_array
  GPUERRORCHECK(cudaMemcpy(arr, gpu_array, size_old, cudaMemcpyDeviceToHost));
  cudaFree(&gpu_array);
}

void swap(int *arr, int fst, int snd) {
	int temp;
	temp = arr[fst];
	arr[fst] = arr[snd];
	arr[snd] = temp;
}

//cpu sequntial version:
extern "C"
void bitonic_sort(int *arr, int sz) {
  size_t size_old = sz * sizeof(int);
  
  int log_new_size = next_power_of_two(arr, sz);
  int size = 1 << log_new_size;

  int *array = (int *) malloc(size * sizeof(int));
  memcpy(array, arr, size_old);

  //now fill the allocated empty cells
  for (int i = sz; i < size; i++) {
    array[i] = INT_MAX;
  }

  for (int step_val = 2; step_val <= size; step_val <<= 1) {
    for (int phase_val = step_val >> 1; phase_val > 0; phase_val >>= 1) {
      for (int num = 0; num < size; num++) {
        int p_num = num ^ phase_val;
        bool is_in_first = (num & step_val) == 0;

        if (p_num < num)
          continue;

        if (is_in_first && array[num] > array[p_num]) 
          swap(array, num, p_num);
        if (!is_in_first && array[num] < array[p_num])
          swap(array, num, p_num);
      }
    }
  }

  memcpy(arr, array, size_old);
}