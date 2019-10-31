#ifndef GAUSS_H_INCLUDED
#define GAUSS_H_INCLUDED

#include <string>

void allocateMemory(int size, double *&b1, double *&b2, double **&mtx1
                            , double **&mtx2, double *&ans1, double *&ans2);

void freeMemory(double *&b1, double *&b2, double **&mtx1
                , double **&mtx2, double *&ans1, double *&ans2);

bool arraysEqual(int size, double *arr1, double *arr2);

void readEquation(std::string f, int size, double **&mtx, double *&b);

void sequentialGauss(double **mtx, int size, double *b, double *&ans);

void parallelGauss(double **mtx, int size, double *b, double *&ans);

void gslGauss(double *mtx, int size, double *b, double *&ans);

void generateMatrices();

#endif

