#include "gauss.h"
#include <omp.h>
#include <gsl/gsl_linalg.h>
#include <utility>
#include <random>
#include <ctime>
#include <vector>
#include <string>
#include <fstream>

using std::vector;
using std::ofstream;
using std::ifstream;
using std::string;
using std::to_string;

void sequentialGauss(double **mtx, int size, double *b, double *&ans) {
    for (int k = 0; k < size - 1; k++) {
        double pivot = mtx[k][k];
        for (int i = k + 1; i < size; i++) {
            double lik = mtx[i][k] / pivot;
            for (int j = k; j < size; j++)
                mtx[i][j] -= lik * mtx[k][j];
            b[i] -= lik * b[k];
        }
    }

    for (int k = size - 1; k >= 0; k--) {
        ans[k] = b[k];
        for (int i = k + 1; i < size; i++)
            ans[k] -= mtx[k][i] * ans[i];
        ans[k] /= mtx[k][k];
    }
}

void parallelGauss(double **mtx, int size, double *b, double *&ans) {
    omp_set_num_threads(omp_get_num_procs());
    for (int k = 0; k < size - 1; k++) {
        double pivot = mtx[k][k];
        for (int i = k + 1; i < size; i++) {
            double lik = mtx[i][k] / pivot;
            #pragma omp simd
            for (int j = k; j < size; j++) {
                mtx[i][j] -= lik * mtx[k][j];
            }
            b[i] -= lik * b[k];
        }
    }

    for (int k = size - 1; k >= 0; k--) {
        ans[k] = b[k];
        #pragma omp simd
        for (int i = k + 1; i < size; i++)
            ans[k] -= mtx[k][i] * ans[i];
        ans[k] /= mtx[k][k];
    }
}


void gslGauss(double *mtx, int size, double *b, double *&ans) {
    gsl_matrix_view gsl_mtx = gsl_matrix_view_array(mtx, size, size);
    gsl_vector_view gsl_b = gsl_vector_view_array(b, size);
    gsl_vector *gsl_ans = gsl_vector_alloc(size);

    int s;

    gsl_permutation *p = gsl_permutation_alloc(size);
    gsl_linalg_LU_decomp(&gsl_mtx.matrix, p, &s);
    gsl_linalg_LU_solve(&gsl_mtx.matrix, p, &gsl_b.vector, gsl_ans);

    for (int i = 0; i < size; ++i) {
        ans[i] = gsl_vector_get(gsl_ans, i);
    }

    gsl_permutation_free(p);
    gsl_vector_free(gsl_ans);
}

bool arraysEqual(int size, double *arr1, double *arr2) {
    for (int i = 0; i < size; ++i) {
        if (abs(arr1[i] - arr2[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

void generateMatrices() {
    vector<int> sizes = {100, 500, 1000, 2000, 3000
                        , 4000, 5000, 10000};

    std:: mt19937 gen(time(0));
    for (int size : sizes) {
        string filename = "bin/equation_" + to_string(size);
        ofstream f(filename.c_str());

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                f << gen() % 100 + 1 << " ";
            }
        }
    }
}

void allocateMemory(int size, double *&b1, double *&b2, double **&mtx1
                            , double **&mtx2, double *&ans1, double *&ans2) {
    b1 = new double [size];
    mtx1 = new double* [size];
    double *tmp1 = new double [size * size];
    for (int i = 0; i < size; ++i) {
        mtx1[i] = tmp1 + i * size;
    }
    b2 = new double [size];
    mtx2 = new double* [size];
    double *tmp2 = new double [size * size];
    for (int i = 0; i < size; ++i) {
        mtx2[i] = tmp2 + i * size;
    }
    ans1 = new double[size];
    ans2 = new double[size];
}

void freeMemory(double *&b1, double *&b2, double **&mtx1
                , double **&mtx2, double *&ans1, double *&ans2) {
    delete[] ans2;
    delete[] ans1;
    delete[] mtx2[0];
    delete[] mtx2;
    delete[] b2;
    delete[] mtx1[0];
    delete[] mtx1;
    delete[] b1;

}

void readEquation(string filename, int size, double **&mtx, double *&b) {
    ifstream f(filename);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            f >> mtx[i][j];
        }
    }

    for (int i = 0; i < size; ++i) {
        f >> b[i];
    }
}

