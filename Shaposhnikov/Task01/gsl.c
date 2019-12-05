#include <gsl/gsl_linalg.h>

#include "imp_gsl.h"

double *gsl_implementation(double* matrix_a, double* vector_b, int size)
{
    gsl_matrix_view gsl_a = gsl_matrix_view_array(matrix_a, size, size);
    gsl_vector_view gsl_b = gsl_vector_view_array(vector_b, size);
    gsl_vector *gsl_x = gsl_vector_alloc(size);
    int s;
    
    gsl_permutation *p = gsl_permutation_alloc(size);
    gsl_linalg_LU_decomp(&gsl_a.matrix, p, &s);
    gsl_linalg_LU_solve(&gsl_a.matrix, p, &gsl_b.vector, gsl_x);
    
    //gsl_permutation_free(p);
    double* solution = malloc(sizeof(*solution) * size);

    for (int i = 0; i < size; ++i) 
        solution[i] = gsl_vector_get(gsl_x, i);

    return solution;
}