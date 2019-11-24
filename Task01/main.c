#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "linear_solve.h"
#include "malloc_check.h"

#define MAX_SIZE 1024
#define NUM_MEASUREMENTS 50
#define STEP 4
#define MAX_DIFF 1e-6

double vectors_max_diff(size_t size, double *expected_v, double *v) {
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
    time_t start;

    for (size_t size = STEP; size <= MAX_SIZE; size *= STEP) {

        double *matrix[size], *matrix_copy[size];
        double *b_vector = malloc_check(size * sizeof(double));

        double *x_vector_seq = malloc_check(size * sizeof(double));
        double *x_vector_omp = malloc_check(size * sizeof(double));
        double *x_vector_expected = malloc_check(size * sizeof(double));
        generate_matrix(size, matrix);
        generate_vector(size, b_vector);

        //Performance measurements
        double gsl_avg_time = 0;
        double seq_avg_time = 0;
        double omp_avg_time = 0;

        for (int time_meas = 0; time_meas < NUM_MEASUREMENTS; time_meas++) {
            int num = 0;
            double *matrix_flat = malloc_check(size * size * sizeof(double));
            for (int i = 0; i < size; i++){
                for (int j = 0; j < size; j++) {
                    matrix_flat[num++] = matrix[i][j];
                }
            }

            //GSL time
            start = clock();
            gsl_solve(size, matrix_flat, b_vector, x_vector_expected);
            gsl_avg_time += (double)(clock() - start) / (CLOCKS_PER_SEC * NUM_MEASUREMENTS);
            free(matrix_flat);

            //OMP time
            double *b_copy = malloc_check(size * sizeof(double));
            copy_sle(size, matrix, b_vector, matrix_copy, b_copy);
            start = clock();
            omp_solve(size, matrix_copy, b_copy, x_vector_omp);
            omp_avg_time += (double)(clock() - start) / (CLOCKS_PER_SEC * NUM_MEASUREMENTS);

            for (int i = 0; i < size; i++) {
                free(matrix_copy[i]);
            }
            free(b_copy);

            //Sequential time
            b_copy = malloc_check(size * sizeof(double));
            copy_sle(size, matrix, b_vector, matrix_copy, b_copy);            
            start = clock();
            sequential_solve(size, matrix_copy, b_copy, x_vector_seq);
            seq_avg_time += (double)(clock() - start) / (CLOCKS_PER_SEC * NUM_MEASUREMENTS);

            for (int i = 0; i < size; i++) {
                free(matrix_copy[i]);
            }
            free(b_copy);
        }

        printf("Test status (0 / 1) sequential: %d omp: %d\n"
            , vectors_max_diff(size, x_vector_expected, x_vector_seq) < MAX_DIFF
            , vectors_max_diff(size, x_vector_expected, x_vector_omp) < MAX_DIFF);

        for (int i = 0; i < size; i++) {
            free(matrix[i]);
        }
        free(b_vector);
        free(x_vector_seq);
        free(x_vector_omp);
        free(x_vector_expected);

        printf("Avg time for        sequential: %f omp: %f gsl: %f | size: %zd\n"
               , seq_avg_time
               , omp_avg_time
               , gsl_avg_time, size);
    }
    
    return 0;
}
