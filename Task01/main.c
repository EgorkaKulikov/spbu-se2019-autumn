#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "test.h"

const int RANDOM_MAX = 999999;
double    PRECISION  = 1e-6;
int       TEST_TIMES = 15;

double* matrix;
double* vectorB;

void checkInput(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Invalid number of arguments (%d)! 2 expected", argc);
        exit(1);
    }

    int  matrixSize = (int) strtol(argv[1], NULL, 10);
    if (matrixSize <= 0)
    {
        printf("Invalid matrix size (%d)!", matrixSize);
        exit(1);
    }
}

void allocateInitMemory(int size)
{
    matrix  = malloc(sizeof(double) * size * size);
    vectorB = malloc(sizeof(double) * size);

    if (matrix == NULL || vectorB == NULL)
    {
        printf("Can't allocate enough memory!");
        exit(1);
    }
}

void generateData(int size)
{
    srand(time(0));
    //Matrix generation
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i * size + j] = (double) (rand() % RANDOM_MAX);

    //Vector generation
    for (int i = 0; i < size; i++)
        vectorB[i] = (double) (rand() % RANDOM_MAX);
}

void freeInitMemory()
{
    free(matrix);
    free(vectorB);
}

int main(int argc, char** argv)
{
    checkInput(argc, argv);

    int matrixSize = (int) strtol(argv[1], NULL, 10);

    allocateInitMemory(matrixSize);
    generateData(matrixSize);

    compareSolutions(matrix, vectorB, matrixSize, PRECISION);
    measureCalculationTime(matrix, vectorB, matrixSize, TEST_TIMES);

    freeInitMemory();
    return 0;
}