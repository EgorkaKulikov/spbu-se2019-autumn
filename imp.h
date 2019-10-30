
#include <iostream>
#include <omp.h>
#include "Eigen/Dense"

using namespace Eigen;

double* consistentImp(double** a, double* b, double* x, int n);
VectorXd eigenImp(MatrixXd a, VectorXd b);
double* parallelImp(double** a, double* b, double* x, int n);
