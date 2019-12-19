#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <iostream>

const int NUM_OF_STATES = 2000;
const int NUM_OF_OBSERVATIONS = 2000;
const int BLOCK_SIZE = 1024;
const int NUM_OF_BLOCK = (NUM_OF_STATES / BLOCK_SIZE) + 1;

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

void sendMatrixToDevice(double** onHost, double* onDevice, int X, int Y)
{
	double** temp = new double* [X * Y];

	for (int i = 0; i < X; i++) {
		memcpy(temp + i * Y, onHost[i], Y * sizeof(double));
	}

	gpuErrCheck(cudaMemcpy(onDevice, temp, X * Y * sizeof(double), cudaMemcpyHostToDevice));

	delete[] temp;
}

void getAndDispose(double** onHost, double* onDevice, int X, int Y)
{
	double** temp = new double* [X * Y];

	gpuErrCheck(cudaMemcpy(temp, onDevice, X * Y * sizeof(double), cudaMemcpyDeviceToHost));

	for (int i = 0; i < X; i++)
		memcpy(onHost[i], temp + i * Y, Y * sizeof(double));

	gpuErrCheck(cudaDeviceSynchronize());

	gpuErrCheck(cudaFree(onDevice));

	delete[] temp;
	
}

__global__
void gpuStep(int* observations, double* transitions, double* emission, double* A, int i)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k < NUM_OF_STATES) {

		A[i * NUM_OF_STATES + k] = 0.0;
		for (int j = 0; j < NUM_OF_STATES; j++) {
			A[i * NUM_OF_STATES + k] += A[(i - 1) * NUM_OF_STATES + j]
				* transitions[j * NUM_OF_STATES + k]
				* emission[k * 2 + observations[i]];
		}
	}
}

void gpuForward(double** A, int* observations, double* distribution, double** transitions, double** emission)
{
	for (int i = 0; i < NUM_OF_STATES; i++) {
		A[0][i] = distribution[i] * emission[i][observations[0]];
	}

	int* gpu_observations;
	double* gpu_A;
	double* gpu_transitions;
	double* gpu_emission;

	gpuErrCheck(cudaMalloc(&gpu_observations, NUM_OF_OBSERVATIONS * sizeof(int)));
	gpuErrCheck(cudaMemcpy(gpu_observations, observations, NUM_OF_OBSERVATIONS * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrCheck(cudaMalloc(&gpu_A, NUM_OF_OBSERVATIONS * NUM_OF_STATES * sizeof(double)));
	gpuErrCheck(cudaMalloc(&gpu_transitions, NUM_OF_STATES * NUM_OF_STATES * sizeof(double)));
	gpuErrCheck(cudaMalloc(&gpu_emission, NUM_OF_STATES * 2 * sizeof(double)));

	gpuErrCheck(cudaPeekAtLastError());

	sendMatrixToDevice(A, gpu_A, NUM_OF_OBSERVATIONS, NUM_OF_STATES);
	sendMatrixToDevice(transitions, gpu_transitions, NUM_OF_STATES, NUM_OF_STATES);
	sendMatrixToDevice(emission, gpu_emission, NUM_OF_STATES, 2);

	gpuErrCheck(cudaDeviceSynchronize());
	
	for (int i = 1; i < NUM_OF_OBSERVATIONS; i++) {
		gpuStep<<<NUM_OF_BLOCK, BLOCK_SIZE>>>(gpu_observations, gpu_transitions, gpu_emission, gpu_A, i);
		gpuErrCheck(cudaPeekAtLastError());
	}

	getAndDispose(A, gpu_A, NUM_OF_OBSERVATIONS, NUM_OF_STATES);

	gpuErrCheck(cudaDeviceSynchronize());

	gpuErrCheck(cudaFree(gpu_observations));
	gpuErrCheck(cudaFree(gpu_transitions));
	gpuErrCheck(cudaFree(gpu_emission));
}

void sequentialStep(int* observations, double** transitions, double** emission, double** A, int i)
{
	for (int j = 0; j < NUM_OF_STATES; j++) {
		A[i][j] = 0.0;

		for (int k = 0; k < NUM_OF_STATES; k++) {
			A[i][j] += A[i - 1][k] * transitions[k][j] * emission[j][observations[i]];
		}
	}
}

void sequentialForward(double** A, int* observations, double* distribution, double** transitions, double** emission)
{
	for (int i = 0; i < NUM_OF_STATES; i++) {
		A[0][i] = distribution[i] * emission[i][observations[0]];
	}

	for (int i = 1; i < NUM_OF_OBSERVATIONS; i++) {
		sequentialStep(observations, transitions, emission, A, i);
	}
}

void initTransitions(double** transitions)
{
	for (int i = 0; i < NUM_OF_STATES; i++)
	{
		double N = 1.0 / NUM_OF_STATES;
		double sum = 0;
		for (int j = 0; j < NUM_OF_STATES; j++)
		{
			sum += N;
			transitions[i][j] = N;
		}

		for (int k = 0; k < NUM_OF_STATES; k++)
		{
			transitions[i][k] /= sum;
		}
	}
}

void initEmission(double** emission)
{
	for (int i = 0; i < NUM_OF_STATES; i++)
	{
		double N = 1.0 / NUM_OF_STATES;
		double sum = 0;
		for (int j = 0; j < 2; j++)
		{
			sum += N;
			emission[i][j] = N;
		}

		for (int k = 0; k < 2; k++)
		{
			emission[i][k] /= sum;
		}
	}
}

void initData(int* observations, double* distribution, double** transitions, double** emission)
{
	srand(time(NULL));

	for (int i = 0; i < NUM_OF_OBSERVATIONS; i++)
	{
		observations[i] = rand() % 2;
	}

	for (int i = 0; i < NUM_OF_STATES; i++)
	{
		distribution[i] = (double)1 / NUM_OF_STATES;
	}

	initTransitions(transitions);
	initEmission(emission);
}

int main()
{
	int* observations = new int[NUM_OF_OBSERVATIONS];

	double* distribution = new double[NUM_OF_STATES];

	double** transitions = new double* [NUM_OF_STATES];
	for (int i = 0; i < NUM_OF_STATES; i++)
	{
		transitions[i] = new double[NUM_OF_STATES];
	}

	double** emission = new double* [NUM_OF_STATES];
	for (int i = 0; i < NUM_OF_STATES; i++)
	{
		emission[i] = new double[2];
	}

	double** A = new double* [NUM_OF_OBSERVATIONS];
	for (int i = 0; i < NUM_OF_OBSERVATIONS; i++) 
	{
		A[i] = new double[NUM_OF_STATES];
	}

	initData(observations, distribution, transitions, emission);

	clock_t from, to;

	from = clock();
	sequentialForward(A, observations, distribution, transitions, emission);
	to = clock();
	printf("Sequential time: %f\n", double(to - from) / CLOCKS_PER_SEC);

	from = clock();
	gpuForward(A, observations, distribution, transitions, emission);
	to = clock();
	printf("GPU time: %f\n", double(to - from) / CLOCKS_PER_SEC);



	delete[] observations;
	delete[] distribution;

	for (int i = 0; i < NUM_OF_STATES; i++) 
	{
		delete[] transitions[i];
		delete[] emission[i];
	}
	delete[] transitions;
	delete[] emission;

	for (int i = 0; i < NUM_OF_OBSERVATIONS; i++)
	{
		delete[] A[i];
	}
	delete[] A;

	return 0;
}

