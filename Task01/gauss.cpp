#include <Eigen/Dense>
#include <omp.h>
#include <utility>
#include <cmath>

#include "gauss.h"

using namespace std;

const int numberOfThreads = omp_get_num_procs();

bool linearGauss(double **matrix, int rows, int cols, vector<double> &ans) {
    int vars = cols - 1;
    int boundVars = (vars < rows) ? vars : rows;
    for (int i = 0; i < boundVars; i++) {
        int maxAbs = i;

        for (int j = i + 1; j < rows; j++) {
            if (abs(matrix[j][i]) > abs(matrix[maxAbs][i])) {
                maxAbs = j;
            }
        }

        swap(matrix[i], matrix[maxAbs]);

        double divider = matrix[i][i];

        if (abs(divider) < epsilon) {
            continue;
        }

        matrix[i][i] = (double)1;

        for (int j = i + 1; j < boundVars; j++) {
            matrix[i][j] /= divider;
        }

        matrix[i][vars] /= divider;

        for (int j = i + 1; j < rows; j++) {
            double multiplier = matrix[j][i];
            matrix[j][i] = (double)0;

            for (int k = i + 1; k < boundVars; k++) {
                matrix[j][k] -= multiplier * matrix[i][k];
            }

            matrix[j][vars] -= multiplier * matrix[i][vars];
        }
    }

    for (int i = boundVars; i < rows; i++) {
        if (abs(matrix[i][vars]) < epsilon) {
            return false;
        }
    }

    ans.assign(vars, (double)0);

    for (int i = boundVars - 1; i >= 0; i--) {
        if (abs(matrix[i][i]) < epsilon) {
            if (abs(matrix[i][vars]) < epsilon) {
                continue;
            }
            else {
                return false;
            }
        }

        double multiplier = matrix[i][vars];
        ans[i] = -multiplier;

        for (int j = i - 1; j >= 0; j--) {
            matrix[j][vars] -= matrix[j][i] * multiplier;
        }
    }

    return true;
}

bool parallelGauss(double **matrix, int rows, int cols, vector<double> &ans) {
    omp_set_num_threads(numberOfThreads);
    int vars = cols - 1;
    int boundVars = (vars < rows) ? vars : rows;
    for (int i = 0; i < boundVars; i++) {
        int maxAbs = i;

        for (int j = i + 1; j < rows; j++) {
            if (abs(matrix[j][i]) > abs(matrix[maxAbs][i])) {
                maxAbs = j;
            }
        }

        swap(matrix[i], matrix[maxAbs]);

        double divider = matrix[i][i];

        if (abs(divider) < epsilon) {
            continue;
        }

        matrix[i][i] = (double)1;

        #pragma omp parallel for
        for (int j = i + 1; j < boundVars; j++) {
            matrix[i][j] /= divider;
        }

        matrix[i][vars] /= divider;

        #pragma omp parallel for
        for (int j = i + 1; j < rows; j++) {
            double multiplier = matrix[j][i];
            matrix[j][i] = (double)0;

            #pragma omp parallel for
            for (int k = i + 1; k < boundVars; k++) {
                matrix[j][k] -= multiplier * matrix[i][k];
            }

            matrix[j][vars] -= multiplier * matrix[i][vars];
        }
    }

    for (int i = boundVars; i < rows; i++) {
        if (abs(matrix[i][vars]) < epsilon) {
            return false;
        }
    }

    ans.assign(vars, (double)0);

    for (int i = boundVars - 1; i >= 0; i--) {
        if (abs(matrix[i][i]) < epsilon) {
            if (abs(matrix[i][vars]) < epsilon) {
                continue;
            }
            else {
                return false;
            }
        }

        double multiplier = matrix[i][vars];
        ans[i] = -multiplier;

        #pragma omp parallel for
        for (int j = i - 1; j >= 0; j--) {
            matrix[j][vars] -= matrix[j][i] * multiplier;
        }
    }

    return true;
}

using namespace Eigen;

bool eigenGauss(double *matrix, double *freeTerms, int rows, int cols, vector<double> &ans) {
    Map<MatrixXd> A = Map<MatrixXd>(matrix, rows, cols);
    A.transposeInPlace(); //this changes the input data but is promised by the Eigen docs to be
                          //more optimized than transpose().eval() which does not
    Map<VectorXd> b = Map<VectorXd>(freeTerms, rows);
    VectorXd x = A.colPivHouseholderQr().solve(b);
    bool hasSolution = (A*x).isApprox(b, epsilon);

    if (hasSolution) {
        for(int i = 0; i < cols; i++) {
            ans.push_back(x[i]);
        }
    }
    return hasSolution;
}

bool verifySolution(double *matrix, double *freeTerms, int rows, int cols, vector<double> &ans) {
    Map<MatrixXd> A = Map<MatrixXd>(matrix, rows, cols);
    Map<VectorXd> b = Map<VectorXd>(freeTerms, rows);
    VectorXd x = VectorXd(cols);

    for (int i = 0; i < cols; i++) {
        x[i] = ans[i];
    }

    return (A*x).isApprox(b, epsilon);
}
