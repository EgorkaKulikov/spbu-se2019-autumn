#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <iostream>

#define STATES_NUM 30
#define OBSERVATIONS_NUM 30
#define RANDOMIZER_ACCURACY 10000
#define BLOCK_SIZE 1024

const int BLOCK_NUM = STATES_NUM / BLOCK_SIZE + 1;

double min4(double a, double b, double c, double d);
void fillInitialDistribution(double* initialDistribution);
void generateObservations(int* observations);
void generateTransitions(double** Transitions);
void generateEmissions(double** Emissions);
int* viterbi(double initialDistribution[], int observations[], double** Transitions, double** Emissions);
int* viterbiGPU(double initialDistribution[], int observations[], double** Transitions, double** Emissions);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) 
			exit(code);
	}
}

int main()
{
	//Initial probability array
	//(the probability of the initial state being S_i)
	double* initialDistribution = new double[STATES_NUM];

	//Sequence of observations
	int* observations = new int[OBSERVATIONS_NUM];

	//Transition probability matrix
	double** Transitions = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		Transitions[i] = new double[STATES_NUM];

	//Probability matrix of observation O_j from state S_i
	double** Emissions = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		Emissions[i] = new double[STATES_NUM];

	// Practice shows that the initial distribution has little effect on
	// therefore make it almost normal
	fillInitialDistribution(initialDistribution);

	generateObservations(observations);

	//Generate double-stochastic transition matrix
	generateTransitions(Transitions);

	//And a column-stochastic emission matrix
	generateEmissions(Emissions);

	clock_t start, end;
	
	start = clock();
	viterbi(initialDistribution, observations, Transitions, Emissions);
	end = clock();
	std::cout << "Viterbi elapsed time: " << double(end - start) / CLOCKS_PER_SEC << " sec\n";

	start = clock();
	viterbiGPU(initialDistribution, observations, Transitions, Emissions);
	end = clock();
	std::cout << "ViterbiGPU elapsed time: " << double(end - start) / CLOCKS_PER_SEC << " sec\n";

	for (int i = 0; i < STATES_NUM; i++)
		delete[]Transitions[i];
	delete[]Transitions;

	for (int i = 0; i < STATES_NUM; i++)
		delete[]Emissions[i];
	delete[]Emissions;

	delete[]initialDistribution;
	delete[]observations;

	return 0;
}

double min4(double a, double b, double c, double d)
{
	double temp[4] = { a, b, c, d };
	double min = DBL_MAX;
	for (int i = 0; i < 4; i++)
		if (temp[i] < min)
			min = temp[i];

	return min;
}

//(Yes, this is probably a crutch monster, but I didn't come up with anything better)
void generateTransitions(double** Transitions)
{
	double n = 1.0 / STATES_NUM;

	for (int i = 0; i < STATES_NUM; i++)
		for (int j = 0; j < OBSERVATIONS_NUM; j++)
			Transitions[i][j] = n;

	srand(time(NULL));

	for (int i = 0; i < STATES_NUM; i++)
	{
		for (int j = 0; j < STATES_NUM; j++)
		{
			int randomIndex = -1;
			do
			{
				randomIndex = rand() % STATES_NUM;
			} while (randomIndex == i || randomIndex == j);

			double maxDelta = min4(Transitions[i][randomIndex],
				Transitions[STATES_NUM - i - 1][STATES_NUM - randomIndex - 1],
				1 - Transitions[i][STATES_NUM - randomIndex - 1],
				1 - Transitions[STATES_NUM - i - 1][randomIndex])
				/ 2;

			//delta - random number from 0 to maxDelta
			double delta = (double)(rand() % (RANDOMIZER_ACCURACY + 1)) / RANDOMIZER_ACCURACY * maxDelta;
			Transitions[i][j] -= delta;
			Transitions[STATES_NUM - i - 1][STATES_NUM - j - 1] -= delta;
			Transitions[i][STATES_NUM - j - 1] += delta;
			Transitions[STATES_NUM - i - 1][j] += delta;
		}
	}
}

void generateEmissions(double** Emissions)
{
	double n = 1.0 / STATES_NUM;

	for (int i = 0; i < STATES_NUM; i++)
		for (int j = 0; j < OBSERVATIONS_NUM; j++)
			Emissions[i][j] = n;

	srand(time(NULL));

	for (int i = 0; i < STATES_NUM; i++)
	{
		for (int j = 0; j < OBSERVATIONS_NUM; j++)
		{
			int randomIndex;
			do
			{
				randomIndex = rand() % STATES_NUM;
			} while (randomIndex == j);

			double min = 0;
			if (Emissions[i][j] < Emissions[i][randomIndex])
				min = Emissions[i][j];
			else
				min = Emissions[i][randomIndex];

			double maxDelta = min / 2;

			//delta - random number from 0 to maxDelta
			double delta = (double)(rand() % (RANDOMIZER_ACCURACY + 1)) / RANDOMIZER_ACCURACY * maxDelta;
			Emissions[i][j] -= delta;
			Emissions[i][randomIndex] += delta;
		}
	}
}

void generateObservations(int* observations)
{
	srand(time(NULL));
	for (int i = 0; i < OBSERVATIONS_NUM; i++)
		observations[i] = rand() % 2;
}

void fillInitialDistribution(double* initialDistribution)
{
	double n = 1.0 / STATES_NUM;
	double sum = 0;
	for (int i = 0; i < STATES_NUM - 1; i++)
	{
		initialDistribution[i] = n;
		sum += n;
	}
	//When dividing, an error can accumulate, so let's do this
	initialDistribution[STATES_NUM - 1] = 1 - sum;
}

//Slightly light version, but it doesn't parallel very well
int* viterbi(double initialDistribution[], int observations[], double** Transitions, double** Emissions)
{
	//Probability matrix for the fact that at the j-th step we are in the state S_i
	double** MState = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		MState[i] = new double[OBSERVATIONS_NUM];

	//Index matrix of the most probable states at j - 1 step
	int** MIndex = new int* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		MIndex[i] = new int[OBSERVATIONS_NUM];

	//Fill the first column based on the initial data
	for (int i = 0; i < STATES_NUM; i++)
	{
		MState[i][0] = initialDistribution[i] * Emissions[i][observations[i]];
		MIndex[i][0] = 0;
	}

	//Fill in the following
	for (int i = 1; i < OBSERVATIONS_NUM; i++) {
		for (int j = 0; j < STATES_NUM; j++) {
			//We are looking for an index at which func is maximized
			int indMax = -1;
			for (int k = 0; k < STATES_NUM; k++)
			{
				double func = MState[k][i - 1] * Transitions[k][j] * Emissions[j][observations[i]];
				if (MState[j][i] < func)
				{
					MState[j][i] = func;
					indMax = k;
				}
			}

			MIndex[j][i] = indMax;
			printf("%d ",indMax);
		}
	}

	double max = -1;
	int* result = new int[OBSERVATIONS_NUM];

	//Select the index of the last state
	for (int i = 0; i < STATES_NUM; i++)
	{
		if (MState[i][OBSERVATIONS_NUM - 1] > max)
		{
			max = MState[i][OBSERVATIONS_NUM - 1];
			result[OBSERVATIONS_NUM - 1] = i;
		}
	}

	//Fill the rest
	for (int i = OBSERVATIONS_NUM - 2; i > 0; i--)
		result[i] = MIndex[result[i + 1]][i + 1];

	for (int i = 0; i < STATES_NUM; i++)
		delete[]MState[i];
	delete[]MState;

	for (int i = 0; i < STATES_NUM; i++)
		delete[]MIndex[i];
	delete[]MIndex;

	return result;
}

__global__
void viterbiGPU_forward(double* MState, double* Transitions, double* Emissions, int* Observations, int i) 
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if (k >= STATES_NUM)
		return;

	MState[i * STATES_NUM + k] = -1;

	for (int j = 0; j < STATES_NUM; j++) 
	{
		double func = MState[(i - 1) * STATES_NUM + j] * Transitions[j * STATES_NUM + k]
					* Emissions[k * 2 + Observations[i]];

		if (MState[i * STATES_NUM + k] < func) 
			MState[i * STATES_NUM + k] = func;
	}
}

__global__
void viterbiGPU_back(double* MState, double* Transitions, double* MIndex, int i) 
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if (k >= STATES_NUM)
		return;

	double max = -1;
	for (int j = 0; j < STATES_NUM; j++) 
	{
		double func = MState[i * STATES_NUM + j] * Transitions[j * STATES_NUM + k];
		if (max < func) 
		{
			max = func;
			MIndex[i * STATES_NUM + k] = j;
		}
	}
}

void copyMatrixFromDevice(double** matrix, double* deviceMatrix, int ROWS_NUM, int COLUMN_NUM) 
{
	double** temp = new double *[ROWS_NUM * COLUMN_NUM];

	gpuErrchk(
		cudaMemcpy(temp, deviceMatrix, ROWS_NUM * COLUMN_NUM * sizeof(double), cudaMemcpyDeviceToHost)
	);

	for (int i = 0; i < ROWS_NUM; i++)
		memcpy(matrix[i], temp + i * COLUMN_NUM, COLUMN_NUM * sizeof(double));
	
	delete[] temp;
}

void copyMatrixToDevice(double **matrix, double *deviceMatrix, int ROWS_NUM, int COLUMN_NUM)
{
	double** temp = new double* [ROWS_NUM * COLUMN_NUM];

	for (int i = 0; i < ROWS_NUM; i++)
		memcpy(temp + i * COLUMN_NUM, matrix[i], COLUMN_NUM * sizeof(double));

	gpuErrchk(
		cudaMemcpy(deviceMatrix, temp, ROWS_NUM * COLUMN_NUM * sizeof(double), cudaMemcpyHostToDevice)
	);

	delete[] temp;
}


int* viterbiGPU(double initialDistribution[], int observations[], double** Transitions, double** Emissions)
{
	double** MState = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		MState[i] = new double[OBSERVATIONS_NUM];

	for (int i = 0; i < STATES_NUM; i++)
		MState[0][i] = initialDistribution[i] * Emissions[i][observations[0]];

	double** MIndex = new double* [OBSERVATIONS_NUM];
	for (int i = 0; i < OBSERVATIONS_NUM; i++)
		MIndex[i] = new double[STATES_NUM];

	int* deviceObservations;
	gpuErrchk(
		cudaMalloc(&deviceObservations, OBSERVATIONS_NUM * sizeof(int))
	);

	double* deviceTransitions;
	gpuErrchk(
		cudaMalloc(&deviceTransitions, STATES_NUM * STATES_NUM * sizeof(double))
	);

	double* deviceEmissions;
	gpuErrchk(
		cudaMalloc(&deviceEmissions, STATES_NUM * STATES_NUM * sizeof(double))
	);

	double* deviceMState;
	gpuErrchk(
		cudaMalloc(&deviceMState, STATES_NUM * OBSERVATIONS_NUM * sizeof(double))
	);

	double* deviceMIndex;
	gpuErrchk(
		cudaMalloc(&deviceMIndex, STATES_NUM * OBSERVATIONS_NUM * sizeof(double))
	);

	gpuErrchk(
		cudaMemcpy(deviceObservations, observations, OBSERVATIONS_NUM * sizeof(int), cudaMemcpyHostToDevice)
	);
	copyMatrixToDevice(Transitions, deviceTransitions, STATES_NUM, STATES_NUM);
	copyMatrixToDevice(Emissions, deviceEmissions, STATES_NUM, STATES_NUM);
	copyMatrixToDevice(MState, deviceMState, STATES_NUM, OBSERVATIONS_NUM);
	copyMatrixToDevice(MIndex, deviceMIndex, STATES_NUM, OBSERVATIONS_NUM);

	gpuErrchk( cudaDeviceSynchronize() );

	for (int i = 1; i < OBSERVATIONS_NUM; i++) 
	{
		viterbiGPU_forward<<<BLOCK_NUM, BLOCK_SIZE>>>
			(deviceMState, deviceTransitions, deviceEmissions, deviceObservations, i);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
	for (int i = 0; i < OBSERVATIONS_NUM; i++)
	{
		viterbiGPU_back <<<BLOCK_NUM, BLOCK_SIZE>>>
			(deviceMState, deviceTransitions, deviceMIndex, i);
		gpuErrchk( cudaPeekAtLastError() );
	}

	copyMatrixFromDevice(MState, deviceMState, OBSERVATIONS_NUM, STATES_NUM);
	copyMatrixFromDevice(MIndex, deviceMIndex, OBSERVATIONS_NUM, STATES_NUM);

	gpuErrchk( cudaFree(deviceMIndex) );
	gpuErrchk( cudaFree(deviceMState) );
	gpuErrchk( cudaFree(deviceEmissions) );
	gpuErrchk( cudaFree(deviceTransitions) );
	gpuErrchk( cudaFree(deviceObservations) );

	double max = -1;
	int* result = new int[OBSERVATIONS_NUM];

	for (int i = 0; i < STATES_NUM; i++)
		if (MState[OBSERVATIONS_NUM - 1][i] > max) 
		{
			max = MState[OBSERVATIONS_NUM - 1][i];
			result[OBSERVATIONS_NUM - 1] = i;
		}


	for (int i = OBSERVATIONS_NUM - 2; i >= 0; i--)
		result[i] = (int)MIndex[i + 1][result[i + 1]];

	for (int i = 0; i < OBSERVATIONS_NUM; i++) 
		delete MIndex[i];
	delete[] MIndex;

	for (int i = 0; i < OBSERVATIONS_NUM; i++)
		delete MState[i];
	delete[] MState;

	return result;
}
