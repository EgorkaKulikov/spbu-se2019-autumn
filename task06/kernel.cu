#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;

#define SIZE_OF_BLOCK 1024

void GenerateData(int states, int countObservations, int* observations, double** transitions,
	double** emission, double* initial_distribution) {
	srand(time(0));
	double sum = 0;

	for (int i = 0; i < countObservations; i++)
		observations[i] = rand() % 2;
	
	//suppose simple generate situation, then probably [i][j] == [i][j + 1] == [i][j + 2]...
	for (int i = 0; i < states; i++) {
		sum = 0;
		for (int j = 0; j < states; j++) {
			transitions[i][j] = 1.0 / states;
			sum += transitions[i][j];
		}
		for (int j = 0; j < states; j++) {
			transitions[i][j] /= sum;
		}
	}

	for (int i = 0; i < states; i++) {
		sum = 0;
		for (int j = 0; j < 2; j++) {
			emission[i][j] = 1.0 / states;
			sum += emission[i][j];
		}
		for (int j = 0; j < 2; j++) {
			emission[i][j] /= sum;
		}
	}
	// Generate initial probabilities distribution (pi)
	sum = 0;
	for (int i = 0; i < states; i++) {
		initial_distribution[i] = 1.0 / states;
		sum += initial_distribution[i];
	}
	for (int i = 0; i < states; i++) {
		initial_distribution[i] /= sum;
	}
}


void ForwardAlgo(int states, int countObservations, int* observations, double** transitions,
	double** emission, double* initial_distribution, double** result) {

	for (int i = 0; i < states; i++) {
		result[0][i] = initial_distribution[i] * emission[i][observations[0]];
	}
	for (int t = 1; t < countObservations; t++) {
		for (int j = 0; j < states; j++) {
			result[t][j] = 0.0;
			for (int i = 0; i < states; i++) {
				result[t][j] += result[t - 1][i] * transitions[i][j] * emission[j][observations[t]];
			}
		}
	}
}

double* allocateMatrixOnDevice(double** matrix, int rows, int column) {
	double* dMatrix;
	const int N = rows * column;
	cudaError_t err;

	if (matrix == NULL) {
		cerr << "allocateMatrixOnDevice: matrix not allocated on host\n";
		exit(1);
	}
	err = cudaMalloc(&dMatrix, sizeof(double) * N);
	if (err != cudaSuccess) {
		cerr << "allocateMatrixOnDevice: cudaMalloc: FAIL\n";
		exit(1);
	}
	

	return dMatrix;
}

void CopyMatrixToDevice(double** matrix, double* dMatrix, int rows, int column) {
	int N = rows * column;
	double** temp = new double* [N];
	cudaError_t err;

	if (matrix == NULL) {
		cerr <<"CopyMatrixToDevice: matrix not allocated on host\n";
		exit(1);
	}

	cudaMemcpy(temp, dMatrix, N * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rows; i++)
		memcpy(temp + i * column, matrix[i], column * sizeof(double));

	err = cudaMemcpy(dMatrix, temp, N * sizeof(double), cudaMemcpyHostToDevice);
	switch (err) {
	case cudaErrorInvalidValue:
		cerr << "CopyMatrixToDevice: cudaMemcpy: InvalidValue\n";
		exit(1);
		break;
	case cudaErrorInvalidDevicePointer:
		cerr << "CopyMatrixToDevice: cudaMemcpy: InvalidDevicePointer\n";
		exit(1);
		break;
	case cudaErrorInvalidMemcpyDirection:
		cerr << "CopyMatrixToDevice: cudaMemcpy: InvalidMemcpyDirection\n";
		exit(1);
		break;
	}

	delete[] temp;

}

void CopyMatrixFromDevice(double **matrix, double *dMatrix, int rows, int column) {
	int N = rows * column;
	double **temp_matrix = new double *[N];

	if (dMatrix == NULL) {
		cerr << "CopyMatrixFromDevice: matrix not allocated on device\n";
		exit(1);
	}

	cudaError_t err;
	err = cudaMemcpy(temp_matrix, dMatrix, N * sizeof(double), cudaMemcpyDeviceToHost);

	switch (err) {
	case cudaErrorInvalidValue:
		cerr <<"CopyMatrixFromDevice: cudaMemcpy: InvalidValue\n";
		exit(1);
		break;
	case cudaErrorInvalidDevicePointer:
		cerr <<"CopyMatrixFromDevice: cudaMemcpy: InvalidDevicePointer\n";
		exit(1);
		break;
	case cudaErrorInvalidMemcpyDirection:
		cerr << "CopyMatrixFromDevice: cudaMemcpy: InvalidMemcpyDirection\n";
		exit(1);
		break;
	}

	for (int i = 0; i < rows; i++) {
		memcpy(matrix[i], temp_matrix + i * column, column * sizeof(double));
	}
	delete[] temp_matrix;
}

__global__
void ForwardStep(double* result, double* transitions, double* emissions, int* observations,
	int t, int states) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < states) {
		result[t * states + i] = 0.0;
		for (int j = 0; j < states; j++) {
			result[t * states + i] += result[(t - 1) * states + j]
				* transitions[j * states + i]
				* emissions[i * 2 + observations[t]];
		}
	}
}


void GPUForwardAlgo(int states, int countObservations, int* observations, double** transitions,
	double** emission, double* initial_distribution, double** result) {
	const int BLOCK_NUMBER = states / SIZE_OF_BLOCK + 1;

	for (int i = 0; i < states; i++) {
		result[0][i] = initial_distribution[i] * emission[i][observations[0]];
	}

	double* dResult = allocateMatrixOnDevice(result, countObservations, states);
	//cudaMalloc(&dResult, countObservations * states * sizeof(double));
	double* dTransitions = allocateMatrixOnDevice(transitions, states, states);
	//cudaMalloc(&dTransitions, states * states * sizeof(double));
	double* dEmissions = allocateMatrixOnDevice(emission, states, 2);
	//cudaMalloc(&dEmissions, states * 2 * sizeof(double));
	int* dObservations;
	cudaMalloc(&dObservations, countObservations * sizeof(int));

	CopyMatrixToDevice(result, dResult, countObservations, states);
	CopyMatrixToDevice(transitions, dTransitions, states, states);
	CopyMatrixToDevice(emission, dEmissions, states, 2);
	cudaMemcpy(dObservations, observations, countObservations * sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	for (int t = 1; t < countObservations; t++) {
		ForwardStep<<<BLOCK_NUMBER, SIZE_OF_BLOCK>>>
			(dResult, dTransitions, dEmissions, dObservations,t, states);
		cudaDeviceSynchronize();
	}

	//copyMatrixFromDevice
	CopyMatrixFromDevice(result, dResult, countObservations, states);

	cudaFree(dObservations);
	cudaFree(dTransitions);
	cudaFree(dEmissions);
	cudaFree(dResult);

}


int main() {
	int const states = 1000;
	int const countObservations = 100;

	//A Transaction Probability Matrix(A) 
	double** transition = new double* [states];  //a[i][j]=p(s(t+1)=j | s(t)=i )
	for (int i = 0; i < states; i++) {
		transition[i] = new double[states];
	}

	//A Emission Probability Matrix (Observation Likelihood)
	double** emission = new double* [states];
	for (int i = 0; i < states; i++) {
		emission[i] = new double[2]; //0/1
		//probability of emitting symbol j given state i.
	}

	//A sequence of T observations (v^T)
	int* observations = new int[countObservations];

	//An Initial Probability Distribution(pi) 
	double* initial_distribution = new double[states];

	//Result Matrix
	double** result = new double* [countObservations];
	for (int i = 0; i < countObservations; i++) {
		result[i] = new double[states];
	}

	GenerateData(states, countObservations, observations, transition, emission, initial_distribution);

	//Run algo
	clock_t start_t, end_t;

	start_t = clock();
	ForwardAlgo(states, countObservations, observations, 
		transition, emission, initial_distribution, result);
	end_t = clock();
	cout <<"Time of consistent forward algo: " <<
		double(end_t - start_t) / CLOCKS_PER_SEC << endl;

	start_t = clock();
	GPUForwardAlgo(states, countObservations, observations,
		transition, emission, initial_distribution, result);
	end_t = clock();
	cout << "Time of GPU parallel forward algo: " <<
		double(end_t - start_t) / CLOCKS_PER_SEC << endl;

	//Deleting arrays
	for (int i = 0; i < states; i++) delete[]transition[i];
	delete[]transition;

	for (int i = 0; i < states; i++) delete[]emission[i];
	delete[]emission;

	delete[]observations;
	delete[]initial_distribution;

	for (int i = 0; i < countObservations; i++) delete[]result[i];
	delete[]result;

	return 0;

}
