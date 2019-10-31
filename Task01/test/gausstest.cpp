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

void GaussTest::testSize10() {
    std::cout << "Test10 is running\n";
    DO_CHECK(passedTest(10));
}

void GaussTest::testSize100() {
    std::cout << "Test100 is running\n";
    DO_CHECK(passedTest(100));
}

void GaussTest::testSize500() {
    std::cout << "Test500 is running\n";
    DO_CHECK(passedTest(500));
}

void GaussTest::testSize1000() {
    std::cout << "Test1000 is running\n";
    DO_CHECK(passedTest(1000));
}

void GaussTest::testSize2000() {
    std::cout << "Test2000 is running";
    DO_CHECK(passedTest(2000));
}

void GaussTest::testSize3000() {
    std::cout << "Test3000 is running";
    DO_CHECK(passedTest(3000));
}

void GaussTest::testSize4000() {
    std::cout << "Test4000 is running";
    DO_CHECK(passedTest(4000));
}

void GaussTest::testSize5000() {
    std::cout << "Test5000 is running\n";
    DO_CHECK(passedTest(5000));
}

void GaussTest::testSize10000() {
    std::cout << "Test10000 is running";
    DO_CHECK(passedTest(10000));
}

void GaussTest::runAllTests() {
    testSize10();
    testSize100();
    testSize500();
    testSize1000();
    testSize2000();
    testSize3000();
    testSize4000();
    testSize5000();
    testSize10000();
    testSize2000();
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
        if (method == "sequential") {
            clock_t start = clock();
            sequentialGauss(mtx1, size, b1, ans1);
            clock_t end = clock();
            times[i] = (end - start) / (double) CLOCKS_PER_SEC;
            averageTime += times[i] * (i != 0); //first launch isn't taken into account
        } else if (method == "parallel") {
            clock_t start = clock();
            parallelGauss(mtx1, size, b1, ans1);
            clock_t end = clock();
            times[i] = (end - start) / (double) CLOCKS_PER_SEC;
            averageTime += times[i] * (i != 0); //first launch isn't taken into account
        } else {
            clock_t start = clock();
            gslGauss(mtx1[0], size, b1, ans1); //first launch isn't taken into account
            clock_t end = clock();
            times[i] = (end - start) / (double) CLOCKS_PER_SEC;
            averageTime += times[i] * (i != 0); 
        }
    }

    averageTime /= (EXPERIMENT_CNT - 1);
    double sum = 0.0;
    for (int i = 1; i < EXPERIMENT_CNT; ++i) {
        sum += (times[i] - averageTime) * (times[i] - averageTime) / (double) (EXPERIMENT_CNT - 2);
    }
    sum = std::sqrt(sum) / std::sqrt(EXPERIMENT_CNT - 1);
    setPrecision(sum, averageTime);
    std::cout << size << ' ' << method << ' ' << averageTime << " +/- " << sum <<'\n';
    freeMemory(b1, b2, mtx1, mtx2, ans1, ans2);
}

void GaussTest::getAllTimes() {
    for (int size : _sizes) {
        for (string method : _methods) {
            getTime(size, method);
        }
    }
}

