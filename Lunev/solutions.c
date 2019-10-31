#include <omp.h>
#include <gsl/gsl_linalg.h>

#include "solutions.h"

#define max_size 100

void generate_matrix(int size, double **matrix, double *RHS)
{
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        RHS [i] = rand() % max_size;
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = 1 + rand() % max_size;
        }
    }
}

void solve_seq(int size, double **matrix,
               double *RHS, double *solutions)
{
    for (int i = 0; i < size-1; i++)
        for (int k = i + 1; k < size; k++)
        {
            double coef = matrix[k][i] / matrix[i][i];
            for (int j = i; j < size; j++)
                matrix[k][j] -= coef * matrix[i][j];
            RHS[k] -= coef * RHS[i];
        }

    for (int i = size-1; i > 0; i--)
        for (int k = i-1; k >= 0; k--)
        {
            double coef = matrix[k][i] / matrix[i][i];
            matrix[k][i] -= matrix[i][i] * coef;
            RHS[k] -= coef * RHS[i];
        }
    for (int i = 0; i < size; i++)
        solutions[i] = RHS[i] / matrix[i][i];
}

void solve_par(int size, double **matrix,
               double *RHS, double *solutions)
{
    for (int i = 0; i < size-1; i++)
        for (int k = i + 1; k < size; k++)
        {
            double coef = matrix[k][i] / matrix[i][i];
            #pragma omp simd
            for (int j = i; j < size; j++)
                matrix[k][j] -= coef * matrix[i][j];
            RHS[k] -= coef * RHS[i];
        }

    for (int i = size-1; i > 0; i--)
    {
        #pragma omp simd
        for (int k = i-1; k >= 0; k--)
        {
            double coef = matrix[k][i] / matrix[i][i];
            matrix[k][i] -= matrix[i][i] * coef;
            RHS[k] -= coef * RHS[i];
        }
    }
    for (int i = 0; i < size; i++)
        solutions[i] = RHS[i] / matrix[i][i];

}


void solve_gsl(int size, double *matrix,
               double *RHS, double *solutions)
{
    gsl_matrix_view gsl_matrix = gsl_matrix_view_array(matrix, size, size);
    gsl_vector_view gsl_RHS = gsl_vector_view_array(RHS, size);
    gsl_vector *gsl_solutions = gsl_vector_alloc(size);
    gsl_permutation *permutation = gsl_permutation_alloc(size);

    int signum;

    gsl_linalg_LU_decomp(&gsl_matrix.matrix, permutation, &signum);
    gsl_linalg_LU_solve(&gsl_matrix.matrix, permutation, &gsl_RHS.vector, gsl_solutions);

    for (int i = 0; i < size; ++i)
    {
        solutions[i] = gsl_vector_get(gsl_solutions, i);
    }

    gsl_permutation_free (permutation);
    gsl_vector_free (gsl_solutions);
}
