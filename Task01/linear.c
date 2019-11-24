#include "linear.h"
#include <gsl/gsl_linalg.h>

void solve_linear(double* initMatrix, double* initVectorB, double* initVectorX, int size)
{
    gsl_matrix_view matrix  = gsl_matrix_view_array(initMatrix,  size, size);
    gsl_vector_view vectorB = gsl_vector_view_array(initVectorB, size);
    gsl_vector*     vectorX = gsl_vector_alloc(size);
    int signum;
    gsl_permutation *p = gsl_permutation_alloc(size);
    gsl_linalg_LU_decomp(&matrix.matrix, p, &signum);
    gsl_linalg_LU_solve (&matrix.matrix, p, &vectorB.vector, vectorX);

    for (int i = 0; i < size; i++)
        initVectorX[i] = gsl_vector_get(vectorX, i);

    gsl_permutation_free(p);
    gsl_vector_free(vectorX);
}