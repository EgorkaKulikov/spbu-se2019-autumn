#pragma once

struct sle {
    double* a;
    double* b;
    int size;
};

extern void solve(struct sle* sle, double* solution);
