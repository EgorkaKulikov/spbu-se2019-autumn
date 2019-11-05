#include <iostream>

#include "sle.h"

int main(void) {
    int size;
    std::cin >> size;

    std::vector<double> a(size * size);
    for (auto& v : a) {
        std::cin >> v;
    }

    std::vector<double> b(size);
    for (auto& v : b) {
        std::cin >> v;
    }

    Sle sle(a, b, size);
    std::vector<double> solution(size);

    sle.solve(solution);
    
    for (auto& v : solution) {
        std::cout << v << std::endl;
    }

    return 0;
}
