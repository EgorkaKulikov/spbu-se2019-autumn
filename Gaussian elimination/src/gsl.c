#include <gsl/gsl_linalg.h>

#include "gsl.h"

void gsl_imp(double* matrix, double* equations, double* solution, int matrix_size) {
    gsl_matrix_view gsl_matrix = gsl_matrix_view_array(matrix, matrix_size, matrix_size);
    gsl_vector_view gsl_equations = gsl_vector_view_array(equations, matrix_size);
    gsl_vector *gsl_solution = gsl_vector_alloc(matrix_size);
    int s;
    gsl_permutation *p = gsl_permutation_alloc(matrix_size);
    gsl_linalg_LU_decomp(&gsl_matrix.matrix, p, &s);
    gsl_linalg_LU_solve(&gsl_matrix.matrix, p, &gsl_equations.vector, gsl_solution);
    for (int i = 0; i < matrix_size; ++i) {
        solution[i] = gsl_vector_get(gsl_solution, i);
    }
}