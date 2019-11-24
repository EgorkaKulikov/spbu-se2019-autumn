#include <omp.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "malloc_check.h"

void copy_sle(size_t size
              , double *matrix[size]
              , double *b
              , double *matrix_copy[size]
              , double *b_copy) {
    for (int i = 0; i < size; i++) {
        matrix_copy[i] = malloc_check(size * sizeof(double));
        memcpy(matrix_copy[i], matrix[i], size * sizeof(double));
    }
    memcpy(b_copy, b, size * sizeof(double));
}

void generate_vector(size_t size, double *b_vector) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        b_vector[i] = rand() % RAND_MAX + 1;;
    }
}

void generate_matrix(size_t size, double *matrix[size]) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        matrix[i] = malloc_check(size * sizeof(double));
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % RAND_MAX + 1;
        }
    }
}

void gsl_solve(size_t size, double *matrix_flat, double *b_vector, double *x_vector) {
    gsl_matrix_view gsl_matrix = gsl_matrix_view_array(matrix_flat, size, size);
    gsl_vector_view gsl_b = gsl_vector_view_array(b_vector, size);
    
    int signum;
    gsl_vector *gsl_x = gsl_vector_alloc(size);
    gsl_permutation *gsl_perm = gsl_permutation_alloc(size);
    gsl_linalg_LU_decomp(&gsl_matrix.matrix, gsl_perm, &signum);
    gsl_linalg_LU_solve(&gsl_matrix.matrix, gsl_perm, &gsl_b.vector, gsl_x);

    for (int i = 0; i < size; i++) {
        x_vector[i] = gsl_vector_get(gsl_x, i);
    }

    gsl_vector_free(gsl_x);
}

void sequential_solve(size_t size, double *matrix[size], double *b_vector, double *x_vector) {
    int pivot_row = 0;
    int pivot_col = 0;
    double multiplier, pivot;

    //forward
    while (pivot_row < size && pivot_col < size) {
        if (matrix[pivot_row][pivot_col] == 0) {
            //no pivot, skip to the next row
            pivot_row++;
        }

        pivot = matrix[pivot_row][pivot_col];
        for (int j = pivot_row + 1; j < size; j++) {
            multiplier = matrix[j][pivot_col] / pivot;
            matrix[j][pivot_col] = 0;
            for (int k = pivot_col + 1; k < size; k++) {
                matrix[j][k] -= multiplier * matrix[pivot_row][k];
            } 
            b_vector[j] -= multiplier * b_vector[pivot_row];
        }

        pivot_row++;
        pivot_col++;
        
    }

    //backwards
    for (int k = size - 1; k >= 0; k--) {
        x_vector[k] = b_vector[k];
        for (int i = k + 1; i < size; i++)
            x_vector[k] -= matrix[k][i] * x_vector[i];
        x_vector[k] /= matrix[k][k];
    }
}

void omp_solve(size_t size, double *matrix[size], double *b_vector, double *x_vector) {
    int pivot_row = 0;
    int pivot_col = 0;
    double multiplier, pivot;

    //forward
    while (pivot_row < size && pivot_col < size) {
        if (matrix[pivot_row][pivot_col] == 0) {
            //no pivot, skip to the next row
            pivot_row++;
        }

        pivot = matrix[pivot_row][pivot_col];    
    
        for (int j = pivot_row + 1; j < size; j++) {
            multiplier = matrix[j][pivot_col] / pivot;
            matrix[j][pivot_col] = 0;
            #pragma omp simd
            for (int k = pivot_col + 1; k < size; k++) {
                matrix[j][k] -= multiplier * matrix[pivot_row][k];
            } 
            b_vector[j] -= multiplier * b_vector[pivot_row];
        }

        pivot_row++;
        pivot_col++;
        
    }

    //backwards
    for (int k = size - 1; k >= 0; k--) {
        x_vector[k] = b_vector[k];
        #pragma omp simd
        for (int i = k + 1; i < size; i++)
            x_vector[k] -= matrix[k][i] * x_vector[i];
        x_vector[k] /= matrix[k][k];
    }
}

