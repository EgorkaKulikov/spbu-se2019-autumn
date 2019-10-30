#ifndef GAUSS_SOLVE
#define GAUSS_SOLVE

void solutionSequental(int n, double **matrix, double *result);

void solutionParallel(int n, double **matrix, double *result);

void solutionWithGSL(int n, double **matrix, double *result);

int getIndex(int a, int b, int n);

#endif