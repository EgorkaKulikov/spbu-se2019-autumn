#include <gsl/gsl_linalg.h>

#include "sle.h"

void Sle::solve(std::vector<double>& solution) {
    gsl_matrix_view a_view = gsl_matrix_view_array(a.data(), size, size);
    gsl_vector_view b_view = gsl_vector_view_array(b.data(), size);
    gsl_vector_view x_view = gsl_vector_view_array(solution.data(), size);

    gsl_permutation* p = gsl_permutation_alloc(size);
    int signum;
    gsl_linalg_LU_decomp(&a_view.matrix, p, &signum);
    gsl_linalg_LU_solve(&a_view.matrix, p, &b_view.vector, &x_view.vector);
}
