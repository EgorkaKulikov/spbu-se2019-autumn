#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "test.h"
#include "sequential.h"
#include "parallel.h"
#include "linear.h"

double compareVectors(double* vectorA, double* vectorB, int size)
{
    double maxDifference = 0;

    for(int i = 0; i < size; i++)
    {
        double currentDifference = fabs(vectorA[i] - vectorB[i]);

        if (currentDifference > maxDifference)
            maxDifference = currentDifference;
    }
}

int compareSolutions(double* matrix, double* vectorB, int size, double precision)
{
    double *vectorX_s = malloc(sizeof(double) * size);
    double *vectorX_p = malloc(sizeof(double) * size);
    double *vectorX_l = malloc(sizeof(double) * size);

    solve_sequential(matrix, vectorB, vectorX_s, size);
    solve_parallel  (matrix, vectorB, vectorX_p, size);
    solve_linear    (matrix, vectorB, vectorX_l, size);

    double difference_s_p = compareVectors(vectorX_s, vectorX_p, size);
    double difference_p_l = compareVectors(vectorX_p, vectorX_l, size);
    double difference_s_l = compareVectors(vectorX_s, vectorX_l, size);

    free(vectorX_s);
    free(vectorX_p);
    free(vectorX_l);

    if (difference_s_p > precision ||
        difference_s_l > precision ||
        difference_p_l > precision)
    {
        printf("Calculations precision less than %f\n", precision);
        return 1;
    }

    printf("Calculations are accurate enough for current precision! (%f)\n", precision);
    return 0;
}

void measureCalculationTime(double* matrix, double* vectorB, int size, int num)
{
    double *vectorX = malloc(sizeof(double) * size);

    double begin = 0, end = 0, total_s = 0, total_p = 0, total_l = 0;

    //Sequential
    for (int i = 0; i < num; i++)
    {
        begin = clock();
        solve_sequential(matrix, vectorB, vectorX, size);
        end = clock();
        total_s = (end - begin) / CLOCKS_PER_SEC;
    }
    total_s /= num;

    //Parallel
    for (int i = 0; i < num; i++)
    {
        begin = clock();
        solve_parallel(matrix, vectorB, vectorX, size);
        end = clock();
        total_p = (end - begin) / CLOCKS_PER_SEC;
    }
    total_p /= num;

    //Linear
    for (int i = 0; i < num; i++)
    {
        begin = clock();
        solve_linear(matrix, vectorB, vectorX, size);
        end = clock();
        total_l += (end - begin) / CLOCKS_PER_SEC;
    }
    total_l /= num;

    free(vectorX);

    printf("Tested %d times. Matrix size is %d^2\n", num, size);
    printf("Average time of sequential solution is %f\n", total_s);
    printf("Average time of parallel (omp) solution is %f\n", total_p);
    printf("Average time of linear (gsl) solution is %f\n", total_l);
}
