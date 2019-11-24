#pragma once

#include <vector>

class Sle {
public:
    std::vector<double>& a;
    std::vector<double>& b;
    int size;

    Sle(std::vector<double>& a, std::vector<double>& b, int size): a(a), b(b), size(size) {}

    void solve(std::vector<double>& solution);
};
