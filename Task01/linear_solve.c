#include <omp.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

void copy_sle(int size
              , double *matrix[size]
              , double *b
              , double *matrix_copy[size]
              , double *b_copy) {
    for (int i = 0; i < size; i++) {
        matrix_copy[i] = malloc(size * sizeof(double));
        memcpy(matrix_copy[i], matrix[i], size * sizeof(double));
    }
    memcpy(b_copy, b, size * sizeof(double));
}

void generate_vector(int size, double *b) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        b[i] = rand() % RAND_MAX + 1;;
    }
}

void generate_matrix(int size, double *matrix[size]) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        matrix[i] = malloc(size * sizeof(double));
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % RAND_MAX + 1;
        }
    }
}

void gsl_solve(int size, double *matrix_flat, double *b, double *x) {
    gsl_matrix_view gsl_matrix = gsl_matrix_view_array(matrix_flat, size, size);
    gsl_vector_view gsl_b = gsl_vector_view_array(b, size);
    
    gsl_vector *gsl_x = gsl_vector_alloc(size);

    gsl_linalg_HH_solve(&gsl_matrix.matrix, &gsl_b.vector, gsl_x);

    for (int i = 0; i < size; i++) {
        x[i] = gsl_vector_get(gsl_x, i);
    }

    gsl_vector_free(gsl_x);
}

void sequential_solve(int size, double *matrix[size], double *b, double *x) {
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
            b[j] -= multiplier * b[pivot_row];
        }

        pivot_row++;
        pivot_col++;
        
    }

    //backwards
    for (int k = size - 1; k >= 0; k--) {
        x[k] = b[k];
        for (int i = k + 1; i < size; i++)
            x[k] -= matrix[k][i] * x[i];
        x[k] /= matrix[k][k];
    }
}

void omp_solve(int size, double *matrix[size], double *b, double *x) {
    omp_set_num_threads(omp_get_num_procs());
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
            b[j] -= multiplier * b[pivot_row];
        }

        pivot_row++;
        pivot_col++;
        
    }

    //backwards
    for (int k = size - 1; k >= 0; k--) {
        x[k] = b[k];
        for (int i = k + 1; i < size; i++)
            x[k] -= matrix[k][i] * x[i];
        x[k] /= matrix[k][k];
    }
}

