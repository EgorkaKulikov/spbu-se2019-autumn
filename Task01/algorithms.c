#include <omp.h>
#include <gsl/gsl_linalg.h>
#include <string.h>

void imp_seq(double *matrix, double *const_vector, double *result_vector, int matrix_size)
{
    for (int k = 0; k < matrix_size - 1; k++)
    {
        double pivot = matrix[k * matrix_size + k]; 
        for (int i = k + 1; i < matrix_size; i++)
        {
            double lik = matrix[i * matrix_size + k] / pivot;
            for (int j = k; j < matrix_size; j++)
                matrix[i * matrix_size + j] -= lik * matrix[k * matrix_size + j];
            const_vector[i] -= lik * const_vector[k];
        }
    }

    for (int k = matrix_size - 1; k >= 0; k--)
    {
        result_vector[k] = const_vector[k];
        for (int i = k + 1; i < matrix_size; i++)
            result_vector[k] -= matrix[k * matrix_size + i] * result_vector[i];
        result_vector[k] /= matrix[k * matrix_size + k];
    }
}

void imp_gsl(double *matrix, double *const_vector, double *result_vector, int matrix_size) {
    gsl_matrix_view gsl_matrix = gsl_matrix_view_array(matrix, matrix_size, matrix_size);
    gsl_vector_view gsl_const_vector = gsl_vector_view_array(const_vector, matrix_size);
    gsl_vector *gsl_result_vector = gsl_vector_alloc(matrix_size);
    int s;
    gsl_permutation *p = gsl_permutation_alloc(matrix_size);

    gsl_linalg_LU_decomp(&gsl_matrix.matrix, p, &s);
    gsl_linalg_LU_solve(&gsl_matrix.matrix, p, &gsl_const_vector.vector, gsl_result_vector);

    for (int i = 0; i < matrix_size; ++i)
    {
        result_vector[i] = gsl_vector_get(gsl_result_vector, i);
    }
}

void imp_par(double *matrix, double *const_vector, double *result_vector, int matrix_size)
{    
    for (int k = 0; k < matrix_size - 1; k++)
    {
        double pivot = matrix[k * matrix_size + k];
        
        for (int i = k + 1; i < matrix_size; i++)
        {
            double lik = matrix[i * matrix_size + k] / pivot;
            
            #pragma omp simd
            for (int j = k; j < matrix_size; j++)
                matrix[i * matrix_size + j] -= lik * matrix[k * matrix_size + j];
            const_vector[i] -= lik * const_vector[k];
        }
    }

    for (int k = matrix_size - 1; k >= 0; k--)
    {
        result_vector[k] = const_vector[k];

        #pragma omp simd
        for (int i = k + 1; i < matrix_size; i++)
            result_vector[k] -= matrix[k * matrix_size + i] * result_vector[i];
        result_vector[k] /= matrix[k * matrix_size + k];
    }
}

