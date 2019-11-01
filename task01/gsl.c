#include "sle.h"

#include <gsl/gsl_linalg.h>

void solve(struct sle* sle, double* solution) {
    gsl_matrix_view a = gsl_matrix_view_array(sle->a, sle->size, sle->size);
    gsl_vector_view b = gsl_vector_view_array(sle->b, sle->size);
    gsl_vector_view x = gsl_vector_view_array(solution, sle->size);

    gsl_permutation *p = gsl_permutation_alloc(sle->size);
    int signum;
    gsl_linalg_LU_decomp(&a.matrix, p, &signum);
    gsl_linalg_LU_solve(&a.matrix, p, &b.vector, &x.vector);
}
