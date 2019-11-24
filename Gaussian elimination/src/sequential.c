#include "sequential.h"

void sequential_imp(double* matrix, double* equations, double* solution, int matrix_size) {
    // Elimination
    for (int k = 0; k < matrix_size; ++k) {
        double pivot = matrix[k * matrix_size + k];
        for (int i = k + 1; i < matrix_size; ++i) {
            double lik = matrix[i * matrix_size + k] / pivot;
            for (int j = k; j < matrix_size; ++j) {
                matrix[i * matrix_size + j] -= lik * matrix[k * matrix_size + j];
            }
            equations[i] -= lik * equations[k];
        }
    }
    // Back substitution
    for (int k = matrix_size - 1; k >= 0; --k) {
        solution[k] = equations[k];
        for (int i = k + 1; i < matrix_size; ++i) {
            solution[k] -= matrix[k * matrix_size + i] * solution[i];
        }
        solution[k] /= matrix[k * matrix_size + k];
    }
}