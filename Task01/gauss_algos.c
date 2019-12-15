#include "gauss_algos.h"

double *GSL_gauss(int n, double *a, double *b) {
    gsl_matrix_view gsl_a = gsl_matrix_view_array(a, n, n);
    gsl_vector_view gsl_b = gsl_vector_view_array(b, n);
    gsl_vector *gsl_x = gsl_vector_alloc(n);
    int s;
    gsl_permutation *p = gsl_permutation_alloc(n);
    gsl_linalg_LU_decomp(&gsl_a.matrix, p, &s);
    gsl_linalg_LU_solve(&gsl_a.matrix, p, &gsl_b.vector, gsl_x);
    gsl_permutation_free(p);
    double *x = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        x[i] = gsl_vector_get(gsl_x, i);
    gsl_vector_free(gsl_x);
    return x;
}

double *seq_gauss(int n, double *a, double *b) {
    double *x = malloc(sizeof(*x) * n);
    // Прямой ход -- O(n^3)
    for (int k = 0; k < n - 1; k++) {
        // Исключение x_i из строк k+1...n-1
        double pivot = a[k * n + k];
        for (int i = k + 1; i < n; i++) {
            // Из уравнения (строки) i вычитается уравнение k
            double heads_ratio = a[i * n + k] / pivot;
            for (int j = k; j < n; j++) {
                a[i * n + j] -= heads_ratio * a[k * n + j];
            }
            b[i] -= heads_ratio * b[k];
        }
    }
    // Обратный ход -- O(n^2)
    for (int k = n - 1; k >= 0; k--) {
        x[k] = b[k];
        for (int i = k + 1; i < n; i++)
            x[k] -= a[k * n + i] * x[i];
        x[k] /= a[k * n + k];
    }
    return x;
}

double *parallel_gauss(int n, double *a, double *b) {
    double *x = malloc(sizeof(*x) * n);
    // Прямой ход -- O(n^3)
    int i, j, k;
    omp_set_num_threads(omp_get_max_threads());
    for (k = 0; k < n - 1; k++) {
        // Исключение x_i из строк k+1...n-1
        double pivot = a[k * n + k];

//#pragma omp parallel for shared(n, a, b, x, pivot, k) private(i, j) default(none) schedule(dynamic)
        for (i = k + 1; i < n; i++) {
            // Из уравнения (строки) i вычитается уравнение k
            double heads_ratio = a[i * n + k] / pivot;
#pragma omp simd
            for (j = k; j < n; j++) {
                a[i * n + j] -= a[k * n + j] * heads_ratio;
            }
            b[i] -= heads_ratio * b[k];
        }
    }

    // Обратный ход -- O(n^2)
    for (k = n - 1; k >= 0; k--) {
        x[k] = b[k];
#pragma omp simd
        for (j = k + 1; j < n; j++)
            x[k] -= a[k * n + j] * x[j];
        x[k] /= a[k * n + k];
    }
    return x;
}

