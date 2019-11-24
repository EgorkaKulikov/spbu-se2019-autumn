#include "gausstest.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <time.h>
#include <cstdio>

using std::memcpy;
using std::string;
using std::to_string;
using std::ifstream;
using std::ofstream;

const int EXPERIMENT_CNT = 10;

void setPrecision(double &err, double &res) {
    int divider = 1;
    while ((int) err == 0) {
        err *= 10;
        divider *= 10;
    }
    res = round(res * divider) / divider;
    err = round(err) / divider;
}

bool GaussTest::passedTest(int size) {    
    double *b1, *b2, **mtx1, **mtx2, *ans1, *ans2;
    allocateMemory(size, b1, b2, mtx1, mtx2, ans1, ans2);

    readEquation("bin/equation_" + to_string(size), size, mtx1, b1);
    memcpy(mtx2[0], mtx1[0], size * size * sizeof(double));
    memcpy(b2, b1, size * sizeof(double));

    sequentialGauss(mtx1, size, b1, ans1);
    gslGauss(mtx2[0], size, b2, ans2);

    bool res = arraysEqual(size, ans1, ans2);
    freeMemory(b1, b2, mtx1, mtx2, ans1, ans2);
    return res;
}

void GaussTest::testSize100() {
    std::cout << "Test100 is running" << std::endl;
    DO_CHECK(passedTest(100));
}

void GaussTest::testSize500() {
    std::cout << "Test500 is running" << std::endl;
    DO_CHECK(passedTest(500));
}

void GaussTest::testSize1000() {
    std::cout << "Test1000 is running" << std::endl;
    DO_CHECK(passedTest(1000));
}

void GaussTest::testSize1500() {
    std::cout << "Test1500 is running" << std::endl;
    DO_CHECK(passedTest(1500));
}

void GaussTest::testSize2000() {
    std::cout << "Test2000 is running" << std::endl;
    DO_CHECK(passedTest(2000));
}

void GaussTest::testSize2500() {
    std::cout << "Test2500 is running" << std::endl;
    DO_CHECK(passedTest(2500));
}

void GaussTest::testSize3000() {
    std::cout << "Test3000 is running" << std::endl;
    DO_CHECK(passedTest(3000));
}

void GaussTest::testSize3500() {
    std::cout << "Test3500 is running" << std::endl;
    DO_CHECK(passedTest(3500));
}

void GaussTest::runAllTests() {
    testSize100();
    testSize500();
    testSize1000();
    testSize1500();
    testSize2000();
    testSize2500();
    testSize3000();
    testSize3500();
}

void GaussTest::getTime(int size, string method) {
    double *b1, *b2, **mtx1, **mtx2, *ans1, *ans2;
    allocateMemory(size, b1, b2, mtx1, mtx2, ans1, ans2);
    readEquation("bin/equation_" + to_string(size), size, mtx1, b1);

    double times[EXPERIMENT_CNT];
    double averageTime = 0.0;

    for (int i = 0; i < EXPERIMENT_CNT; ++i) {
        memcpy(mtx2[0], mtx1[0], size * size);
        memcpy(b2, b1, size);
        if (method[0] == 's') {
            clock_t start = clock();
            sequentialGauss(mtx1, size, b1, ans1);
            clock_t end = clock();
            times[i] = (end - start) / (double) CLOCKS_PER_SEC;
            averageTime += times[i] * (i == 0 ? 0 : 1); //first launch isn't taken into account
        } else if (method[0] == 'p') {
            clock_t start = clock();
            parallelGauss(mtx1, size, b1, ans1);
            clock_t end = clock();
            times[i] = (end - start) / (double) CLOCKS_PER_SEC;
            averageTime += times[i] * (i == 0 ? 0 : 1); //first launch isn't taken into account
        } else {
            clock_t start = clock();
            gslGauss(mtx1[0], size, b1, ans1);
            clock_t end = clock();
            times[i] = (end - start) / (double) CLOCKS_PER_SEC;
            averageTime += times[i] * (i == 0 ? 0 : 1); //first launch isn't taken into account
        }
    }

    averageTime /= (EXPERIMENT_CNT - 1);
    double sum = 0.0;
    for (int i = 1; i < EXPERIMENT_CNT; ++i) {
        sum += (times[i] - averageTime) * (times[i] - averageTime);
    }
    double err = std::sqrt(sum / ((double) (EXPERIMENT_CNT - 2) * (EXPERIMENT_CNT - 1)));
    setPrecision(err, averageTime);
    std::cout << size << ' ' << method << ' ' << averageTime << " +/- " << err << std::endl;
    freeMemory(b1, b2, mtx1, mtx2, ans1, ans2);
}

void GaussTest::getAllTimes() {
    for (int size : _sizes) {
        for (string method : _methods) {
            getTime(size, method);
        }
    }
}

