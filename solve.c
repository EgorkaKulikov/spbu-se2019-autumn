#include <omp.h>
#include <gsl/gsl_linalg.h>

int getIndex(int first, int second, int n) {
    return first * n + second;
}

void solutionWithGSL(int n, double *matrix, double *vector, double *result) {
    gsl_matrix_view gslMatrix = gsl_matrix_view_array(matrix, n, n);
    gsl_vector_view gslVector = gsl_vector_view_array(vector, n);
    gsl_vector *gslResult = gsl_vector_alloc(n);
    gsl_permutation *perm = gsl_permutation_alloc(n);

    int _;
    gsl_linalg_LU_decomp(&gslMatrix.matrix, perm, &_);
    gsl_linalg_LU_solve(&gslMatrix.matrix, perm, &gslVector.vector, gslResult);

    for (int i = 0; i < n; i++) {
        result[i] = gsl_vector_get(gslResult, i);
    }

    gsl_vector_free(gslResult);
    gsl_permutation_free(perm);
}

void solutionSequental(int n, double *matrix, double *vector, double *result) {
    for (int cur = 0; cur < n; cur++) {
        double pivot = matrix[getIndex(cur, cur, n)];
        for (int i = cur + 1; i < n; i++) {       
            double mult = matrix[getIndex(i, cur, n)] / pivot;
            vector[i] -= mult * vector[cur];
            for (int j = cur; j < n; j++) {
                matrix[getIndex(i, j, n)] -= mult * matrix[getIndex(cur, j, n)];
            }
        }
    }

    for (int cur = n - 1; cur >= 0; cur--) {
        result[cur] = vector[cur] / matrix[getIndex(cur, cur, n)];
        for (int i = cur - 1; i >= 0; i--) {
            vector[i] -= result[cur] * matrix[getIndex(i, cur, n)];
        }
    }
}

void solutionParallel(int n, double *matrix, double *vector, double *result) {
    omp_set_num_threads(omp_get_num_procs());

    for (int cur = 0; cur < n; cur++) {
        double pivot = matrix[getIndex(cur, cur, n)];
        #pragma omp parallel for
        for (int i = cur + 1; i < n; i++) {       
            double mult = matrix[getIndex(i, cur, n)] / pivot;
            vector[i] -= mult * vector[cur];
            for (int j = cur; j < n; j++) {
                matrix[getIndex(i, j, n)] -= mult * matrix[getIndex(cur, j, n)];
            }
        }
    }

    for (int cur = n - 1; cur >= 0; cur--) {
        result[cur] = vector[cur] / matrix[getIndex(cur, cur, n)];
        #pragma omp simd
        for (int i = cur - 1; i >= 0; i--) {
            vector[i] -= result[cur] * matrix[getIndex(i, cur, n)];
        }
    }
}
