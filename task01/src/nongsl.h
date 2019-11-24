#pragma once

#include "sle.h"

class Computer {
public:
    Sle& sle;
    std::vector<int> permutation;
    int step;

    Computer(Sle& sle);

    void do_step();
    void prepare();
    void compute_solution(std::vector<double>& solution);

    double* get_a(int line) {
        return sle.a.data() + permutation[line] * sle.b.size();
    }

    double& get_b(int line) {
        return sle.b[permutation[line]];
    }
};
