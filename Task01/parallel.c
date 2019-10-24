#include "parallel.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void solve_parallel(double* initMatrix, double* initVectorB, double* vectorX, int size)
{
    double* matrix  = malloc(sizeof(double) * size * size);
    double* vectorB = malloc(sizeof(double) * size);

    memcpy(matrix,  initMatrix,  sizeof(double) * size * size);
    memcpy(vectorB, initVectorB, sizeof(double) * size);

    omp_set_num_threads(omp_get_num_procs());

    for (int k = 0; k < size - 1; k++)
    {
        double pivot = matrix[k * size + k];

        for (int i = k + 1; i < size; i++)
        {
            double multiplier = matrix[i * size + k] / pivot;

            #pragma omp simd

            for (int j = k; j < size; j++)
                matrix[i * size + j] -= multiplier * matrix[k * size + j];

            vectorB[i] -= multiplier * vectorB[k];
        }
    }

    for (int k = size - 1; k >= 0; k--)
    {
        vectorX[k] = vectorB[k];

        #pragma omp simd

        for (int i = k + 1; i < size; i++)
            vectorX[k] -= matrix[k * size + i] * vectorX[i];

        vectorX[k] /= matrix[k * size + k];
    }

    free(matrix);
    free(vectorB);
}