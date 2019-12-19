#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <cstdio>
#include <random>
#include <time.h>
#include <iostream>

using namespace std;

const int THREADS_NUM = 1024;
const int EXPRMNT_NUM = 10;

void bitonic_sort(int* arr, int size) 
{
  int num_stages = 0;
  //log2 size floored
  for (int i = size; i > 0; i >>= 1, num_stages++);
  num_stages--;

  for (int stage = 1; stage <= num_stages; stage++) 
  {
    int num_passes = stage;
    int block_size = 1 << stage;
    int num_blocks = size >> stage;
    for (int pass = 0; pass < num_passes; pass++) 
    {
      int step = block_size >> 1;
      for (int block = 0; block < num_blocks; block++) 
      {
        bool ascending = ((block >> pass) & 1) == 0;
        for (int i = 0; i < step; i++) 
        {
          int index = block * block_size + i;
          if ((ascending && (arr[index] > arr[index + step])) 
          || (!ascending && (arr[index] < arr[index + step]))) 
          {
            int tmp = arr[index];
            arr[index] = arr[index + step];
            arr[index + step] = tmp;
          }
        }
      }
      block_size >>= 1;
      num_blocks <<= 1;
    }
  }
  cout << endl;
}

__device__
void swap_gpu(int* array, int first, int second)
{   
    int tmp = array[first];
    array[first] = array[second];
    array[second] = tmp;
}

__global__
void bitonic_exchange_gpu(int* dev_values, int block_size, unsigned long stage)
{
    unsigned int i, match; 

    i = threadIdx.x + blockDim.x * blockIdx.x;

    match = i + (block_size >> 1);

    bool ascending_i = (i & (1 << stage)) == 0;
    bool ascending_m = (match & (1 << stage)) == 0;

    if (ascending_i ^ ascending_m == 1) return;
    
    if ((dev_values[i] > dev_values[match]) == ascending_i)
    {
      swap_gpu(dev_values, i, match);
    }
}

void bitonic_sort_CUDA(int* arr, int size) 
{
  int* cudarr;
  if (cudaMalloc(&cudarr, size * sizeof(int)) != cudaSuccess) 
  {
    cerr << "Error when allocating device memory" << endl;
    exit(7);
  }

  if (cudaMemcpy(cudarr, arr, size * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) 
  {
    cerr << "Error when copying memory" << endl;
    exit(8);
  }

  dim3 threadsPerBlock = (size < THREADS_NUM) ? 1 : THREADS_NUM;
  dim3 numCUDABlocks = (size < THREADS_NUM) ? size : size / THREADS_NUM;

  int num_stages = -1;
  for (int i = size; i > 0; i >>= 1, num_stages++);

  for (int stage = 1; stage <= num_stages; stage++) 
  {
    int num_passes = stage;
    int block_size = 1 << stage;
    for (int pass = 1; pass <= num_passes; pass++) 
    {
      bitonic_exchange_gpu <<< numCUDABlocks, threadsPerBlock >>> (cudarr, block_size, stage);
      cudaError_t errSync = cudaGetLastError();
      cudaError_t errAsync = cudaDeviceSynchronize();
      if (errSync != cudaSuccess && errAsync != cudaSuccess) 
      {
        cerr << "CUDA execution error" << endl;
        exit(9);
      }
      block_size >>= 1;
    }
  }

  if (cudaMemcpy(arr, cudarr, size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) 
  {
    cerr << "Error when copying memory" << endl;
    exit(8);
  }

  cudaFree(cudarr);
}

int main(int argc, char **argv) 
{
  if (argc != 3) 
  {
    cerr << "Wrong number of arguments" << endl;
    exit(1);
  }

  FILE* file = fopen(argv[1], "r");
  if (file == nullptr) 
  {
    cerr << "No such file of directory" << endl;
    exit(2);
  }

  int size;
  if (fscanf(file, "%d", &size) != 1) 
  {
    cerr << "Wrong file format" << endl;
    exit(3);
  }

  if (size == 0) 
  {
    return 0;
  }

  int power_of_2_size = 1;
  while (power_of_2_size < size) 
  {
    power_of_2_size <<= 1;
  }

  int* data = new int[power_of_2_size];
  if (data == nullptr) 
  {
    cerr << "Memory allocation error" << endl;
    exit(4);
  }

  int max_value = (1 << 30);
  for (int i = 0; i < size; i++) 
  {
    if (fscanf(file, "%d", &(data[i])) != 1) 
    {
      cerr << "Error while reading file!" << endl;
      return 3;
    }
    max_value = max(max_value, data[i]);
  }

  fclose(file);

  for (int i = size; i < power_of_2_size; i++) 
  {
    data[i] = max_value;
  }

  double time_avg = 0.0;
  for (int counter = 0; counter < EXPRMNT_NUM; counter++)
  {
    int* data_copy = (int*) malloc(power_of_2_size * sizeof(int));
    memcpy(data_copy, data, power_of_2_size * sizeof(int));
    double time_count = -1.0;
    if (argv[2][0] == 's') 
    {
      double timestamp = clock();
      bitonic_sort(data_copy, power_of_2_size);
      time_count = (double) (clock() - timestamp);
    }
    else if (argv[2][0] == 'p') 
    {
      double timestamp = clock();
      bitonic_sort_CUDA(data_copy, power_of_2_size);
      time_count = (double) (clock() - timestamp);
    }
    else 
    {
      cerr << "Wrong launch parameter" << endl;
      exit(5);
    }

    int prev = data_copy[0];
    //check if sorted
    for (int i = 1; i < power_of_2_size; i++) 
    {
      if (data_copy[i] < prev) 
      {
        cerr << "error at element: " << i << "data size: " << size << "changed size: " << power_of_2_size << "prev value: " << prev << "next value: " << data_copy[i] << endl;
      }
      prev = data_copy[i];
    }
    time_avg += time_count;
    free(data_copy);
  }
  cout << "Time: " << (time_avg / EXPRMNT_NUM) / CLOCKS_PER_SEC << "s, " << size << " elements" << endl;
  delete[] data;
  return 0;
}