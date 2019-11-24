#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include "Eigen/Dense"

#include "sequentialGauss.h"
#include "parallelGauss.h"
#include "eigenGauss.h"

using namespace std;

typedef long double ld;

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Expected 2 parameters.";
        return 1;
    }

    int n;
    cin >> n;
    vector<vector<ld>> matrix_A(n, vector<ld> (n));
    vector<ld> vector_b(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            cin >> matrix_A[i][j];
        cin >> vector_b[i];
    }

    if (!strcmp(argv[1], "sequential")) {
        vector<vector<ld>> matrix_Ab_T(n+1, vector<ld> (n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                matrix_Ab_T[j][i] = matrix_A[i][j];
            matrix_Ab_T[n][i] = vector_b[i];
        }

        clock_t begin_sg = clock();
        vector<ld> sx = sequentialGauss(matrix_Ab_T);
        clock_t end_sg = clock();

        cout << ld(end_sg - begin_sg) / CLOCKS_PER_SEC << endl;
        ofstream fout("sx.txt");
        for (int i = 0; i < n; i++)
            fout << sx[i] << endl;
    }
    else if (!strcmp(argv[1], "parallel")) {
        vector<vector<ld>> matrix_Ab_T(n+1, vector<ld> (n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                matrix_Ab_T[j][i] = matrix_A[i][j];
            matrix_Ab_T[n][i] = vector_b[i];
        }

        clock_t begin_pg = clock();
        vector<ld> px = parallelGauss(matrix_Ab_T);
        clock_t end_pg = clock();

        cout << ld(end_pg - begin_pg) / CLOCKS_PER_SEC << endl;
        ofstream fout("px.txt");
        for (int i = 0; i < n; i++)
            fout << px[i] << endl;
    }
    else if (!strcmp(argv[1], "eigen")) {
        Eigen::MatrixXd eigen_matrix_A(n, n);
        Eigen::VectorXd  eigen_vector_b(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                eigen_matrix_A(i, j) = matrix_A[i][j];
            eigen_vector_b(i) = vector_b[i];
        }

        clock_t begin_eg = clock();
        Eigen::VectorXd ex = eigenGauss(eigen_matrix_A, eigen_vector_b);
        clock_t end_eg = clock();

        cout << ld(end_eg - begin_eg) / CLOCKS_PER_SEC << endl;
        ofstream fout("ex.txt");
        for (int i = 0; i < n; i++)
            fout << ex(i) << endl;
    }
    else {
        cerr << "Unknown second parameter.";
        return 1;
    }
    return 0;
}
