#include <omp.h>

#include "parallel.h"

void parallel_imp(double* matrix, double* equations, double* solution, int matrix_size) {

    int num_procs = omp_get_num_procs() / 2;
    omp_set_num_threads(num_procs);

    omp_set_schedule(omp_sched_auto, 0);

    for (int k = 0; k < matrix_size; ++k) {
        double pivot = matrix[k * matrix_size + k];
        #pragma omp parallel for default(shared)
        for (int i = k + 1; i < matrix_size; ++i) {
            double lik = matrix[i * matrix_size + k] / pivot;
            #pragma omp simd
            for (int j = k; j < matrix_size; ++j) {
                matrix[i * matrix_size + j] -= lik * matrix[k * matrix_size + j];
            }
            equations[i] -= lik * equations[k];
        }
    }

    for (int k = matrix_size - 1; k >= 0; --k) {
        solution[k] = equations[k];
        #pragma omp simd
        for (int i = k + 1; i < matrix_size; ++i) {
            solution[k] -= matrix[k * matrix_size + i] * solution[i];
        }
        solution[k] /= matrix[k * matrix_size + k];
    }
}
