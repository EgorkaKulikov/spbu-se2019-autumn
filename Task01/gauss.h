#ifndef GAUSS
#define GAUSS

#include <vector>

using namespace std;

bool linearGauss(double **matrix, int rows, int cols, vector<double> &ans);
bool parallelGauss(double **matrix, int rows, int cols, vector<double> &ans);
bool eigenGauss(double *matrix, double *freeTerms, int rows, int cols, vector<double> &ans);
bool verifySolution(double *matrix, double *freeTerms, int rows, int cols, vector<double> &ans);

const double epsilon = 0.000001;

#endif // GAUSS
