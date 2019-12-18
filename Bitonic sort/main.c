#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "gpu.h"

#define MAX_VALUE 65536

void printElapsed(clock_t start, clock_t stop) {
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.5fs\n", elapsed);
}

bool isSorted(int *arr, int n) {
    for (int i = 1; i < n; ++i)
        if (arr[i - 1] > arr[i])
            return false;
    return true;
}

bool isEqual(int *foo, int *bar, int n) {
    for (int i = 0; i < n; ++i)
        if (foo[i] != bar[i])
            return false;
    return true;
}

void valuesFill(int *arr, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; ++i) arr[i] = rand() % MAX_VALUE;
}

int upperBin(int n) {
    int r = 1;
    while (r < n) r <<= 1;
    return r;
}

void compare(int* a, int* b) {
    int ta = *a, tb = *b;
    if (ta > tb) *a = tb, *b = ta;
}

void bitonicSortCpuStep(int* values, int n) {
    int K = log2(n), d = 1 << K; --K;
    for (int i = 0; i < d >> 1; ++i)
        compare(&values[i], &values[d - i - 1]);
    for (int k = K; k > 0; k--) {
        d = 1 << k;
        for (int j = 0; j < n; j += d)
            for (int i = 0; i < d >> 1; ++i)
                compare(&values[j + i], &values[j + (d >> 1) + i]);
    }
}

void bitonicSortCpu(int* values, int n) {
    int *temp = (int*)malloc(n * sizeof(int));
    memcpy(temp, values, n * sizeof(int));
    int K = log2(n);
    for (int k = 1, d = 2; k <= K; ++k, d <<= 1)
        for (int i = 0; i < n; i += d)
            bitonicSortCpuStep((int*)&temp[i], d);
    memcpy(values, temp, n * sizeof(int));
    free(temp);
    return;
}

int main(int argc, char **argv)
{   
    if (argc != 2) {
        printf("Usage: %s <array size>\n", argv[0]);
        exit(1);
    }

    int numVals = upperBin(atoi(argv[1]));

    clock_t start, stop;

    int *hostValues = (int*)malloc(numVals * sizeof(int)),
        *hostValuesForCpu = (int*)malloc(numVals * sizeof(int));

    valuesFill(hostValues, numVals);
    memcpy(hostValuesForCpu, hostValues, numVals * sizeof(int));

    start = clock();
    bitonicSortGpu(hostValues, numVals);
    stop = clock();

    printElapsed(start, stop);

    if (isSorted(hostValues, numVals))
        printf("Gpu sorting correctly\n");
    else
        printf("Gpu sorting incorrectly\n");

    start = clock();
    bitonicSortCpu(hostValuesForCpu, numVals);
    stop = clock();

    printElapsed(start, stop);

    if (isSorted(hostValuesForCpu, numVals))
        printf("Cpu sorting correctly\n");
    else
        printf("Cpu sorting incorrectly\n");

    if (isEqual(hostValues, hostValuesForCpu, numVals))
        printf("Gpu values are equal to cpu values\n");
    else
        printf("Gpu values are not equal to Cpu values\n");

    free(hostValuesForCpu);
    free(hostValues);

    return 0;
}