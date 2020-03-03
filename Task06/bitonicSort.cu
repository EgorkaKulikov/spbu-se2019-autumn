#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#define MAX_THREADS 512
#define MAX_VALUE 1 << 16

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char*file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void swap(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = *a;
}

__device__
void swapGpu(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = *a;
}

void bitonicStepCpu(int* vals, int n) {
	int logn = log2(n);
	int d = 1 << logn;
	--logn;
	for (int i = 0; i < d >> 1; ++i) {
		if (vals[i] > vals[d - i - 1])
			swap(&vals[i], &vals[d - i - 1]);
	}
	for (int k = logn; k > 0; --k) {
		d = 1 << k;
		for (int j = 0; j < n; j += d)
			for (int i = 0; i < d >> 1; ++i) {
				if (vals[i + j] > vals[i + j + (d >> 1)])
					swap(&vals[i + j], &vals[i + j + (d >> 1)]);
			}
	}
}

void bitonicSortCpu(int* vals, int n) {
	int* tmp = (int *)malloc(n * sizeof(int));
	memcpy(tmp, vals, n * sizeof(int));
	int logn = log2(n);
	for (int k = 1, d = 2; k <= logn; ++k, d <<= 1)
		for (int i = 0; i < n; i += d)
			bitonicStepCpu((int *)&tmp[i], d);
	memcpy(vals, tmp, n * sizeof(int));
	free(tmp);
	return;
}

__global__ void bitonicStepGpu(int *deviceValues, int j, int k) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int xor = i ^ j;
	if (k > i) {
		if ((i & k) == 0) {
			if (deviceValues[i] > deviceValues[xor])
				swapGpu(&deviceValues[i], &deviceValues[xor]);
		} else {
			if (deviceValues[i] < deviceValues[xor])
				swapGpu(&deviceValues[i], &deviceValues[xor]);
		}
	}
}

extern "C"
void bitonicSortGpu(int *vals, int valsCnt) {
	int *deviceValues;
	size_t size = valsCnt * sizeof(int);

	gpuErrCheck(cudaMalloc((void**)&deviceValues, size));
	gpuErrCheck(cudaMemcpy(deviceValues, vals, size, cudaMemcpyHostToDevice))

	int numThreads;
	int numBlocks;

	if (valsCnt <= MAX_THREADS) {
		numThreads = valsCnt;
		numBlocks = 1;
	} else {
		numThreads = MAX_THREADS;
		numBlocks = valsCnt / numThreads;
	}

	dim3 blocks(numBlocks, 1);
	dim3 threads(numThreads, 1);

	for (int k = 2; k <= valsCnt; k <<= 1) {
		for (int j = k >> 1; j > 0; j >>= 1) {
			bitonicStepGpu <<<blocks, threads>>> (deviceValues, j, k);
			gpuErrCheck(cudaGetLastError());
		}
	}

	gpuErrCheck(cudaMemcpy(vals, deviceValues, size, cudaMemcpyDeviceToHost));
	gpuErrCheck(cudaFree(deviceValues));
}

double printTime(char *type, clock_t start, clock_t stop) {
	double time = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("%s time: %.5fs\n", type, time);
	return time;
}

bool isSorted(int *vals, int n) {
	for (int i = 0; i < n - 1; ++i)
		if (vals[i] > vals[i + 1])
			return false;
	return true;
}

void generateMatrix(int *vals, int n) {
	srand(time(NULL));
	for (int i = 0; i < n; ++i)
		vals[i] = rand() % MAX_VALUE;
}

int main() {
	double cpuTime[25], gpuTime[25];
	int sizes[25];
	for (int i = 0, valsCnt = 4; valsCnt < 2 << 25; valsCnt = valsCnt << 1, ++i) {
		printf("Size = %d\n", valsCnt);
		sizes[i] = valsCnt;
		clock_t start, stop;

		int *gpuVals = (int *)malloc(valsCnt * sizeof(int));
		int	*cpuVals = (int *)malloc(valsCnt * sizeof(int));

		generateMatrix(gpuVals, valsCnt);
		memcpy(cpuVals, gpuVals, valsCnt * sizeof(int));

		start = clock();
		bitonicSortGpu(gpuVals, valsCnt);
		stop = clock();

		gpuTime[i] = printTime("GPU", start, stop);

		start = clock();
		bitonicSortCpu(cpuVals, valsCnt);
		stop = clock();

		cpuTime[i] = printTime("CPU", start, stop);

		free(cpuVals);
		free(gpuVals);
	}
	return 0;
}