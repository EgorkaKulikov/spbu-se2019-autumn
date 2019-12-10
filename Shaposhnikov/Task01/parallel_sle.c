#include <omp.h>
#include <stdlib.h> 
#include "imp_parallel.h"

double *parallel_sle(double* matrix, double* vector_b, int matrix_size)
{
    double *answer = malloc(sizeof(*answer) * matrix_size);

	for (int k = 0; k < matrix_size - 1; k++)
	{
		double pivot = matrix[k * matrix_size + k];
		for (int i = k + 1; i < matrix_size; i++)
		{
			double lik = matrix[i * matrix_size + k] / pivot;
			#pragma omp for simd
			for (int j = k; j < matrix_size; j++)
				matrix[i * matrix_size + j] -= lik * matrix[k * matrix_size + j];
			vector_b[i] -= lik * vector_b[k];
		}
	}
	for (int k = matrix_size - 1; k >= 0; k--)
	{
		answer[k] = vector_b[k];
		for (int i = k + 1; i < matrix_size; i++)
			answer[k] -= matrix[k * matrix_size + i] * answer[i];
		answer[k] /= matrix[k * matrix_size + k];
	}
	return answer;
}