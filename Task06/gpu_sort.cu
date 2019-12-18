#define MAX_DEPTH  16
#define MIN_LENGHT 32

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define GPUERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__
void selection_sort(int *array, int left, int right) {

  for (int i = left; i <= right; ++i) {

    int min_value = array[i];
    int min_id    = i;

    for (int j = i + 1; j <= right; ++j) {

      if (array[j] < min_value) {
      
        min_value = array[j];
        min_id    = j;
      }
    }

    if (i != min_id) {

      array[min_id] = array[i];
      array[i]     = min_value;
    }
  }
}

__global__
void quicksort(int *array, int left, int right, int depth){
  
  if (MAX_DEPTH <= depth || MIN_LENGHT >= right - left) {
  
    selection_sort(array, left, right);
    return;
  }

  cudaStream_t s, s1;

  int* left_ptr  = array + left;
  int* right_ptr = array + right;

  int pivot = array[(left + right) / 2];

  int left_value, right_value;

  int new_right, new_left;

  while (left_ptr <= right_ptr) {
  
    left_value  = *left_ptr;
    right_value = *right_ptr;

    while (left_value < pivot && left_ptr < array + right) {
    
      left_ptr++;
      left_value = *left_ptr;
    }

    while (right_value > pivot && right_ptr > array + left) {
    
      right_ptr--;
      right_value = *right_ptr;
    }

    if (left_ptr <= right_ptr) {
    
      *left_ptr  = right_value;
      *right_ptr = left_value;
      
      left_ptr++;
      right_ptr--;
    }
  }

  new_right = right_ptr - array;
  new_left  = left_ptr - array;

  if (left < right_ptr - array) {
    
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(array, left, new_right, depth + 1);
    cudaStreamDestroy(s);
  }

  if (left_ptr - array < right) {
    
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s1>>>(array, new_left, right, depth + 1);
    cudaStreamDestroy(s1);
  }
}

extern "C"
void gpu_sort(int *array, int size){

  int* gpu_array;
  int left = 0;
  int right = size - 1;

  GPUERRCHK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));
  
  GPUERRCHK(cudaMalloc((void**) &gpu_array, size * sizeof(int)));
  GPUERRCHK(cudaMemcpy(gpu_array, array, size * sizeof(int), cudaMemcpyHostToDevice));
  
  quicksort<<<1, 1>>>(gpu_array, left, right, 0);
  GPUERRCHK(cudaDeviceSynchronize());
  
  GPUERRCHK(cudaMemcpy(array, gpu_array, size*sizeof(int), cudaMemcpyDeviceToHost));
  GPUERRCHK(cudaFree(gpu_array));
  
  GPUERRCHK(cudaDeviceReset());
}
