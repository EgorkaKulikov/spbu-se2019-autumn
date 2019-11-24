#include <stdio.h>
#include <time.h>
#include <malloc.h>
#include <stdlib.h>
#include <omp.h>
#include <gsl/gsl_linalg.h>
#include "solution.h"

double* sequential(int n, double *a, double *b) {

    double *x = malloc(sizeof(*x) * n); 

    for (int k = 0; k < n - 1; k++) {
        double pivot = a[k * n + k];
        for (int i = k + 1; i < n; i++) {
            double lik = a[i * n + k] / pivot;
            for (int j = k; j < n; j++)
                a[i * n + j] -= lik * a[k * n + j];
            b[i] -= lik * b[k];
        }   
    }
    
    for (int k = n - 1; k >= 0; k--) { 
        x[k] = b[k];
        for (int i = k + 1; i < n; i++) 
            x[k] -= a[k * n + i] * x[i];
        x[k] /= a[k * n + k]; 
    }

    return x;
}

double* gsl(int n, double *a, double *b) {

    double *x = malloc(sizeof(*x) * n); 

    gsl_matrix_view gsl_a = gsl_matrix_view_array(a, n, n);
    gsl_vector_view gsl_b = gsl_vector_view_array(b, n);
    gsl_vector *gsl_x = gsl_vector_alloc(n);
    int s;
    gsl_permutation *p = gsl_permutation_alloc(n);
    gsl_linalg_LU_decomp(&gsl_a.matrix, p, &s);
    gsl_linalg_LU_solve(&gsl_a.matrix, p, &gsl_b.vector, gsl_x);
 
    for (int i = 0; i < n; i++)
       x[i] = gsl_vector_get(gsl_x, i);

    gsl_permutation_free(p);
    gsl_vector_free(gsl_x);

    return x;
}

double* parallel(int n, double *a, double *b) {

    double *x = malloc(sizeof(*x) * n); 

    omp_set_num_threads(omp_get_num_procs());

    for (int k = 0; k < n - 1; k++) {
        double pivot = a[k * n + k];
            for (int i = k + 1; i < n; i++) {
                double lik = a[i * n + k] / pivot;
                #pragma omp simd
                for (int j = k; j < n; j++)
                    a[i * n + j] -= lik * a[k * n + j];
                b[i] -= lik * b[k];
            }   
        }

    for (int k = n - 1; k >= 0; k--) { 
        x[k] = b[k];
        #pragma omp simd
        for (int i = k + 1; i < n; i++) 
            x[k] -= a[k * n + i] * x[i];
        x[k] /= a[k * n + k]; 
    }

    return x;
}
