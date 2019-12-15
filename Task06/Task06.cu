#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

int INF = INT_MAX;
const int BLOCK_NUM = 262144; //this const allow to work with big amounts of data, but arrays with length less then it will not pass

#define GPUERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void bitonic_sort_step(int *a, int j, int i) {
  unsigned int k, m;
  k = threadIdx.x + blockDim.x * blockIdx.x;
  m = k^j;

  if (m>k) {
    if ((i&k)==0) {
      if (a[k]>a[m]) {
        int temp = a[k];
        a[k] = a[m];
        a[m] = temp;
      }
    }
    if ((i&k)!=0) {
      if (a[k]<a[m]) {
        int temp = a[k];
        a[k] = a[m];
        a[m] = temp;
      }
    }
  }
}

void bitonicSort(int *l, int n) {
  int *a;
  size_t size = n * sizeof(int);
  GPUERRCHK(cudaMalloc((void**) &a, size));
  GPUERRCHK(cudaMemcpy(a, l, size, cudaMemcpyHostToDevice));

  dim3 numBlocks(BLOCK_NUM, 1);
  dim3 numThreads(n/BLOCK_NUM, 1);

  int i, j;
  for (i = 2; i <= n; i <<= 1) {
    for (j=i>>1; j>0; j=j>>1)
      bitonic_sort_step<<<numBlocks, numThreads>>>(a, j, i);
  }
  GPUERRCHK(cudaMemcpy(l, a, size, cudaMemcpyDeviceToHost));
  cudaFree(a);
}

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Invalid number of arguments\n");
		return 1;
	}
	
	int n = 1, i = 0;

	FILE* input = fopen(argv[1], "r");
	if (input == 0) {
		printf("Error in opening file\n");
		return 2;
	}

	int sz;
	if (fscanf(input, "%d", &sz) != 1) {
		printf("Error while reading file 1\n");
		return 3;
	}

	int* nums;
	while (n < sz) n *= 2;
	nums = (int*)malloc(n * sizeof(int));

	for (i = 0; i < sz; i++) {
		if (fscanf(input, "%d", nums + i) != 1) {
			printf("Error while reading file 2\n");
			return 3;
		}
	}	

	fclose(input);
	i = sz;
	while (i < n) nums[i++] = INF;
	
	clock_t startTime = clock();
	bitonicSort(nums, n);
	clock_t endTime = clock();
	double totalTime = (double) (endTime - startTime) / CLOCKS_PER_SEC / 20;
    
    //test
	short test = 1;
	for(i = 1; i<sz; i++){
		if(nums[i]<nums[i-1]) test = 0;
	}
	if(test == 0){
		printf("Array is not sorted");
		return 1;
	}
    
	printf("Time CPU: %f\n", totalTime);
	return 0;
}
