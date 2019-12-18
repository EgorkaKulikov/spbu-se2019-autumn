#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS 1024

__global__ void bitonicSortGpuStep(int *deviceValues, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x, ixj = i ^ j;

    if (ixj > i) {
        if ((i&k) == 0) {
            if (deviceValues[i] > deviceValues[ixj]) {
                int temp = deviceValues[i];
                deviceValues[i] = deviceValues[ixj];
                deviceValues[ixj] = temp;
            }
        } else {
            if (deviceValues[i] < deviceValues[ixj]) {
                int temp = deviceValues[i];
                deviceValues[i] = deviceValues[ixj];
                deviceValues[ixj] = temp;
            }
        }
    }
}

extern "C"
void bitonicSortGpu(int *hostValues, int numVals)
{
    int *deviceValues;
    size_t size = numVals * sizeof(int);

    cudaError_t cudaError = cudaMalloc((void**) &deviceValues, size);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cannot allocate Gpu memory for deviceValues: %s\n",
                cudaGetErrorString(cudaError));
        exit(1);
    }

    cudaError = cudaMemcpy(deviceValues, hostValues, size, cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cannot copy data from hostValues to deviceValues: %s\n",
                cudaGetErrorString(cudaError));
        exit(1);
    }

    int numThreads;
    int numBlocks;

    if (numVals <= MAX_THREADS) {
        numThreads = numVals;
        numBlocks = 1;
    } else {
        numThreads = MAX_THREADS;
        numBlocks = numVals / numThreads;
    }

    dim3 blocks(numBlocks, 1);
    dim3 threads(numThreads, 1);

    for (int k = 2; k <= numVals; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortGpuStep<<<blocks, threads>>>(deviceValues, j, k);

            cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                        cudaGetErrorString(cudaError));
                exit(1);
            }
        }
    }

    cudaError = cudaMemcpy(hostValues, deviceValues, size, cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cannot copy data from deviceValues to hostValues: %s\n",
                cudaGetErrorString(cudaError));
        exit(1);
    }

    cudaError = cudaFree(deviceValues);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Cannot free Gpu memory for deviceValues: %s\n",
                cudaGetErrorString(cudaError));
        exit(1);
    }
}