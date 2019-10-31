#include "gauss.h"
#include <utility>
#include <string>
#include <iostream>
#include <fstream>

using std::string;
using std::to_string;
using std::cin;
using std::cout;
using std::ofstream;
using std::ifstream;
using std::exit;

int main(int argc, char **argv) {
    //generateMatrices();
    if (argc != 4) {
        cout << "Wrong number of arguments!\n";
        exit(2);
    }

    int size = atoi(argv[1]);
    if (size <= 0) {
        cout << "Invalid size of matrix!\n";
        exit(3);
    }

    string method = argv[2];
    if (method != "sequential" || method != "parallel" || method != "gsl") {
        cout << "Unknown method!\n";
        exit(3);
    }

    string filename = argv[3];
    ifstream f;
    f.open(filename.c_str());
    if (!f.is_open()) {
        cout << "Cannot open file " << filename << "\n";
        exit(4);
    }

    double *b = new double [size];
    if (b == nullptr) {
        cout << "Not enough memory for vector b!\n";
        exit(1);
    }

    double **mtx = new double* [size];
    if (mtx == nullptr) {
        cout << "Not enough memory for matrix!\n";
        exit(1);
    }

    double *tmp = new double [size * size];
    if (tmp == nullptr) {
        cout << "Not enough memory for matrix!\n";
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        mtx[i] = tmp + i * size;
    }

    double *ans = new double[size];
    if (ans == nullptr) {
        cout << "Not enough memory for answer!\n";
        exit(1);
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            f >> mtx[i][j];
        }
    }
    for (int i = 0; i < size; ++i) {
        f >> b[i];
    }

    if (method == "sequential") {
        sequentialGauss(mtx, size, b, ans);
    } else if (method == "parallel") {
        parallelGauss(mtx, size, b, ans);
    } else {
        gslGauss(mtx[0], size, b, ans);
    }

    for (int i = 0; i < size; i++) {
        printf("%f\n", ans[i]);
    }
    printf("\n");

    delete[] b;
    delete[] mtx[0];
    delete[] mtx;
    delete[] ans;
}

