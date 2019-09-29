#include <cstdio>
#include<cstdlib>
#include <vector>
#include <ctime>

#include "gauss.h"
#include "test.h"
#include "errors.h"

using namespace std;

void test(const char *fileName) {
    FILE *input = fopen(fileName, "r");

    if (input == nullptr) {
        finish(fileErrorCode);
    }

    toClose.push_back(input);

    int rows, cols;

    if (fscanf(input, "%d%d", &rows, &cols) != 2) {
        finish(inputErrorCode);
    }

    double *parallelMemory = new (std::nothrow) double[rows * cols];
    toCleanUp.push_back(parallelMemory);
    double **parallelMatrix = new (std::nothrow) double *[rows];
    toCleanUp.push_back(parallelMatrix);
    double *linearMemory = new (std::nothrow) double[rows * cols];
    toCleanUp.push_back(linearMemory);
    double **linearMatrix = new (std::nothrow) double *[rows];
    toCleanUp.push_back(linearMatrix);
    double *eigenMatrix = new (std::nothrow) double[rows * (cols - 1)];
    toCleanUp.push_back(eigenMatrix);
    double *eigenFreeTerms= new (std::nothrow) double[rows];
    toCleanUp.push_back(eigenFreeTerms);

    if (parallelMemory == nullptr
        || parallelMatrix == nullptr
        || linearMemory == nullptr
        || linearMatrix == nullptr
        || eigenMatrix == nullptr
        || eigenFreeTerms == nullptr) {
        finish(memoryErrorCode);
    }

    double *parallelRowStart = parallelMemory;
    double *linearRowStart = linearMemory;

    for (int i = 0; i < rows; i++) {
        parallelMatrix[i] = parallelRowStart;
        parallelRowStart += cols;
        linearMatrix[i] = linearRowStart;
        linearRowStart += cols;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(input, "%lf", &(parallelMatrix[i][j])) != 1) {
                finish(inputErrorCode);
            }

            linearMatrix[i][j] = parallelMatrix[i][j];

            if (j == cols - 1) {
                eigenFreeTerms[i] = -parallelMatrix[i][j];
            }
            else {
                eigenMatrix[i * (cols - 1) + j] = parallelMatrix[i][j];
            }
        }
    }

    vector<double> parallelAns;
    vector<double> linearAns;
    vector<double> eigenAns;

    int time_elapsed = clock();
    bool parallelHasSolution = parallelGauss(parallelMatrix, rows, cols, parallelAns);
    int new_time = clock();
    printf("Parallel solution has spent %lf seconds\n", (double)(new_time - time_elapsed) / CLOCKS_PER_SEC);
    time_elapsed = new_time;
    bool linearHasSolution = linearGauss(linearMatrix, rows, cols, linearAns);
    new_time = clock();
    printf("Linear solution has spent %lf seconds\n", (double)(new_time - time_elapsed) / CLOCKS_PER_SEC);
    time_elapsed = new_time;
    bool eigenHasSolution = eigenGauss(eigenMatrix, eigenFreeTerms, rows, cols - 1, eigenAns);
    new_time = clock();
    printf("Eigen solution has spent %lf seconds\n", (double)(new_time - time_elapsed) / CLOCKS_PER_SEC);
    time_elapsed = new_time;

    if (eigenHasSolution) {
        for (double i: eigenAns) {
            printf("%lf ", i);
        }

        printf("\n");

        if (!verifySolution(eigenMatrix, eigenFreeTerms, rows, cols-1, parallelAns)) {
            printf("Parallel function has found a wrong solution\n");

            for (double i: parallelAns) {
                printf("%lf ", i);
            }

            printf("\n");
        }

        if (!verifySolution(eigenMatrix, eigenFreeTerms, rows, cols-1, linearAns)) {
            printf("Linear function has found a wrong solution\n");

            for (double i: linearAns) {
                printf("%lf ", i);
            }

            printf("\n");
        }
    }
    else {
        printf("No solution!\n");

        if (parallelHasSolution) {
            printf("Parallel function has wrongly found a solution\n");
        }

        if (linearHasSolution) {
            printf("Linear function has wrongly found a solution\n");
        }
    }
}
