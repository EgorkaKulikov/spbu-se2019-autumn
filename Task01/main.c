#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "linear_solve.h"

#define MAX_SIZE 1024
#define NUM_MEASUREMENTS 100
#define STEP 4
#define MAX_DIFF 1e-6

double vectors_max_diff(int size, double *expected_v, double *v) {
    double max = 0;
    double diff;
    for (int i = 0; i < size; i++){
        diff = fabs(expected_v[i] - v[i]);
        if (diff > max) {
            max = diff;
        }
    }
    return max;
}

int main(int argc, char *argv[]) {
    time_t start, end;

    for (int size = STEP; size <= MAX_SIZE; size *= STEP) {

        double *matrix[size], *matrix_copy[size];
        double *b = malloc(size * sizeof(double));

        double *x_seq = malloc(size * sizeof(double));
        double *x_omp = malloc(size * sizeof(double));
        double *x_expected = malloc(size * sizeof(double));
        generate_matrix(size, matrix);
        generate_vector(size, b);

        //Performance measurements

        double gsl_avg_time = 0;
        double seq_avg_time = 0;
        double omp_avg_time = 0;

        for (int time_meas = 0; time_meas < NUM_MEASUREMENTS; time_meas++) {
            start = clock();
            int num = 0;
            double *matrix_flat = malloc(size * size * sizeof(double));
            for (int i = 0; i < size; i++){
                for (int j = 0; j < size; j++) {
                    matrix_flat[num++] = matrix[i][j];
                }
            }

            gsl_solve(size, matrix_flat, b, x_expected);
            gsl_avg_time += (double)(clock() - start) / (CLOCKS_PER_SEC * NUM_MEASUREMENTS);
            free(matrix_flat);

            double *b_copy = malloc(size * sizeof(double));
            copy_sle(size, matrix, b, matrix_copy, b_copy);
            start = clock();
            omp_solve(size, matrix_copy, b_copy, x_omp);
            omp_avg_time += (double)(clock() - start) / (CLOCKS_PER_SEC * NUM_MEASUREMENTS);

            for (int i = 0; i < size; i++) {
                free(matrix_copy[i]);
            }
            free(b_copy);

            b_copy = malloc(size * sizeof(double));
            copy_sle(size, matrix, b, matrix_copy, b_copy);            
            start = clock();
            sequential_solve(size, matrix_copy, b_copy, x_seq);
            seq_avg_time += (double)(clock() - start) / (CLOCKS_PER_SEC * NUM_MEASUREMENTS);

            for (int i = 0; i < size; i++) {
                free(matrix_copy[i]);
            }
            free(b_copy);
        }

        printf("Test status (0 / 1) sequential: %d omp: %d\n"
            , vectors_max_diff(size, x_expected, x_seq) < MAX_DIFF
            , vectors_max_diff(size, x_expected, x_omp) < MAX_DIFF);

        for (int i = 0; i < size; i++) {
            free(matrix[i]);
        }
        free(b);
        free(x_seq);
        free(x_omp);
        free(x_expected);


        printf("Avg time for        sequential: %f omp: %f gsl: %f | size: %d\n"
               , seq_avg_time
               , omp_avg_time
               , gsl_avg_time, size);
    }
    
    return 0;
}
