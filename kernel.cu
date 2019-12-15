#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <iostream>

#define STATES_NUM 2000
#define OBSERVATIONS_NUM 2000
#define RANDOMIZER_ACCURACY 10000
#define BLOCK_SIZE 1024

//Почти всегда == 1, т.к. много состояний не лезет в память (при BLOCK_SIZE == 1024)
const int BLOCK_NUM = STATES_NUM / BLOCK_SIZE + 1;

double min4(double a, double b, double c, double d);
void fillInitialDistribution(double* initialDistribution);
void generateObservations(int* observations);
void generateTransitions(double** Transitions);
void generateEmissions(double** Emissions);
int* viterbi(double initialDistribution[], int observations[], double** Transitions, double** Emissions);
int* viterbiGPU(double initialDistribution[], int observations[], double** Transitions, double** Emissions);

int main()
{
	//Массив начальных вероятностей
	//(вероятность того, в начальный момент времени состояние было S_i)
	double* initialDistribution = new double[STATES_NUM];

	//Последовательность наблюдений
	int* observations = new int[OBSERVATIONS_NUM];

	//Матрица вероятностей перехода из i-го состояния в j-тое
	double** Transitions = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		Transitions[i] = new double[STATES_NUM];

	//Матрица вероятностей наблюдения O_j из состояния S_i
	double** Emissions = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		Emissions[i] = new double[STATES_NUM];

	//Как показывает практика, начальное распределение мало на что влияет,
	//поэтому сделаем его нормальным (ну почти)
	fillInitialDistribution(initialDistribution);

	generateObservations(observations);

	//Сгенерируем дважды стохастическую матрицу переходов
	generateTransitions(Transitions);

	//И стохастическую по столбцам матрицу эмиссии
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

//(Да, это, наверное, костыльный монстр, но я не придумал ничего лучше)
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

			//delta - случайное чило от 0 до maxDelta
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

			//delta - случайное чило от 0 до maxDelta
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
	//При делении может накапливаться погрешность, поэтому сделаем так
	initialDistribution[STATES_NUM - 1] = 1 - sum;
}

//Слегка облегчённая версия, но параллелится не очень
int* viterbi(double initialDistribution[], int observations[], double** Transitions, double** Emissions)
{
	//Матрица вероятностей того, что на j-том шаге мы находимся в состоянии S_i
	double** MState = new double* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		MState[i] = new double[OBSERVATIONS_NUM];

	//Матрица индексов наиболее вероятных состояний на j - 1 шаге
	int** MIndex = new int* [STATES_NUM];
	for (int i = 0; i < STATES_NUM; i++)
		MIndex[i] = new int[OBSERVATIONS_NUM];

	//Заполняем первый столбец на основе начальных данных
	for (int i = 0; i < STATES_NUM; i++)
	{
		MState[i][0] = initialDistribution[i] * Emissions[i][observations[i]];
		MIndex[i][0] = 0;
	}

	//Заполняем последующие
	for (int i = 1; i < OBSERVATIONS_NUM; i++) {
		for (int j = 0; j < STATES_NUM; j++) {
			//Ищем индекс, при котором максимизируется func
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
		}
	}

	double max = -1;
	int* result = new int[OBSERVATIONS_NUM];

	//Вносим индекс последнего состояния
	for (int i = 0; i < STATES_NUM; i++)
	{
		if (MState[i][OBSERVATIONS_NUM - 1] > max)
		{
			max = MState[i][OBSERVATIONS_NUM - 1];
			result[OBSERVATIONS_NUM - 1] = i;
		}
	}

	//Заполняем остальные
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

//Дальше комментов не будет( я усталь

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

	cudaMemcpy(temp, deviceMatrix, ROWS_NUM * COLUMN_NUM * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < ROWS_NUM; i++)
		memcpy(matrix[i], temp + i * COLUMN_NUM, COLUMN_NUM * sizeof(double));
	
	delete[] temp;
}

void copyMatrixToDevice(double **matrix, double *deviceMatrix, int ROWS_NUM, int COLUMN_NUM)
{
	double** temp = new double* [ROWS_NUM * COLUMN_NUM];

	for (int i = 0; i < ROWS_NUM; i++)
		memcpy(temp + i * COLUMN_NUM, matrix[i], COLUMN_NUM * sizeof(double));
	cudaMemcpy(deviceMatrix, temp, ROWS_NUM * COLUMN_NUM * sizeof(double), cudaMemcpyHostToDevice);

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
		cudaMalloc(&deviceObservations, OBSERVATIONS_NUM * sizeof(int));

	double* deviceTransitions;
		cudaMalloc(&deviceTransitions, STATES_NUM * STATES_NUM * sizeof(double));

	double* deviceEmissions;
		cudaMalloc(&deviceEmissions, STATES_NUM * STATES_NUM * sizeof(double));

	double* deviceMState;
		cudaMalloc(&deviceMState, STATES_NUM * OBSERVATIONS_NUM * sizeof(double));

	double* deviceMIndex;
		cudaMalloc(&deviceMIndex, STATES_NUM * OBSERVATIONS_NUM * sizeof(double));

	cudaMemcpy(deviceObservations, observations, OBSERVATIONS_NUM * sizeof(int), cudaMemcpyHostToDevice);
	copyMatrixToDevice(Transitions, deviceTransitions, STATES_NUM, STATES_NUM);
	copyMatrixToDevice(Emissions, deviceEmissions, STATES_NUM, STATES_NUM);
	copyMatrixToDevice(MState, deviceMState, STATES_NUM, OBSERVATIONS_NUM);
	copyMatrixToDevice(MIndex, deviceMIndex, STATES_NUM, OBSERVATIONS_NUM);

	cudaDeviceSynchronize();

	for (int i = 1; i < OBSERVATIONS_NUM; i++) 
	{
		viterbiGPU_forward<<<BLOCK_NUM, BLOCK_SIZE>>>
			(deviceMState, deviceTransitions, deviceEmissions, deviceObservations, i);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < OBSERVATIONS_NUM; i++) 
		viterbiGPU_back<<<BLOCK_NUM, BLOCK_SIZE>>>
			(deviceMState, deviceTransitions, deviceMIndex, i);

	copyMatrixFromDevice(MState, deviceMState, OBSERVATIONS_NUM, STATES_NUM);
	copyMatrixFromDevice(MIndex, deviceMIndex, OBSERVATIONS_NUM, STATES_NUM);

	cudaFree(deviceMIndex);
	cudaFree(deviceMState);
	cudaFree(deviceTransitions);
	cudaFree(deviceEmissions);
	cudaFree(deviceObservations);

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
