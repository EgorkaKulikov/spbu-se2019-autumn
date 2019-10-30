#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include "solve.h"

#define false 0
#define true 1
#define EPS ((double)0.0001)

enum solutions
{
    GSL,
    PARALLEL,
    SEQUENTAL,
};

#define EXP_LIST_SIZE 10
int exp_list[EXP_LIST_SIZE] = {10, 25, 50, 100, 175, 250, 325, 500, 750, 1000};

void checkNull(_Bool expr) {
    if (!expr) {
        printf("task failed. Possible cause: No memory.\n");
        exit(1);
    }
}

double *copy(int size, double *array) {
    double *copied = malloc(sizeof(double) * size);
    checkNull(copied);
    memcpy(copied, array, sizeof(double) * size);
    return copied;
}

void createLinearSystem(int n, double *matrix, double *vector) {
    srand(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[getIndex(i, j, n)] = rand() % 1000;
        }
        vector[i] = (double)(rand() % 1000 + 1);
    }
}

_Bool compare(double *first, double *second, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(first[n] - second[n]) > EPS)
            return false;
    }
    return true;
}

double countTime(void (*func)(), int n, double *matrix, double *vector, double *result) {
    double *matrix_copy = copy(n * n, matrix);
    double *vector_copy = copy(n, vector);
    checkNull(matrix_copy && vector_copy);

    clock_t start = clock();
    func(n, matrix_copy, vector_copy, result);
    clock_t end = clock();

    free(matrix_copy);
    free(vector_copy);
    return ((double)end - start) / CLOCKS_PER_SEC;
}

void printSqMatrix(int sz, double *array) {
    for (int j = 0; j < sz; j++) {
        for (int i = 0; i< sz; i++) {
            printf("%.1f ", array[getIndex(i, j, sz)]);
        }
        printf("\n");
    }
}

void printArray(int sz, double *array) {
    for (int i = 0; i < sz; i++) {
        printf("%.1f ", array[i]);
    }
    printf("\n");
}

int main() {
    for (int i = 0; i < EXP_LIST_SIZE; i++) {
        int size = exp_list[i];
        
        double *matrix           = malloc(sizeof(double) * size * size);
        double *vector           = malloc(sizeof(double) * size);
        double *resultGSL        = malloc(sizeof(double) * size);
        double *result_parallel  = malloc(sizeof(double) * size);
        double *result_sequental = malloc(sizeof(double) * size);

        checkNull(matrix 
                   && vector 
                   && resultGSL 
                   && result_parallel 
                   && result_sequental
                 ); 

        printf("Test #%d with the matrix of size {%d x %d}\n", i + 1, size, size);

        createLinearSystem(size, matrix, vector);

        double timeGSL        = countTime(solutionWithGSL, size, matrix, vector, resultGSL);
        double time_sequental = countTime(solutionSequental, size, matrix, vector, result_sequental);
        double time_parallel  = countTime(solutionParallel, size, matrix, vector, result_parallel);

        printf("     GSL: %.6f\n", timeGSL);
        printf("     SEQ: %.6f\n", time_sequental);
        printf("     PAR: %.6f\n", time_parallel);

        printf("\n   with the following result: ");
        _Bool seq_res = compare(resultGSL, result_sequental, size);
        _Bool par_res = compare(resultGSL, result_parallel, size);
        if (seq_res && par_res)
            printf("Correct answer!\n\n\n");
        else if (!(seq_res && par_res))
            printf("Sequental and Parallel answers are both wrong\n\n\n");
        else if (!seq_res)
            printf("Sequental answer is wrong\n\n\n");
        else 
            printf("Parallel answer is wrong\n\n\n");

        free(matrix);
        free(vector);
        free(resultGSL);
        free(result_sequental);
        free(result_parallel);
    }
}