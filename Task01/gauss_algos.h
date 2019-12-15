#ifndef TASK01_GAUSS_ALGOS_H
#define TASK01_GAUSS_ALGOS_H
    #pragma once
    #include <omp.h>
    #include <stdlib.h>
    #include <gsl/gsl_linalg.h>
    double *GSL_gauss(int n, double *a, double *b);

    double *seq_gauss(int n, double *a, double *b);

    double *parallel_gauss(int n, double *a, double *b);

#endif //TASK01_GAUSS_ALGOS_H
