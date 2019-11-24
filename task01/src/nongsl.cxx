#include <string.h>

#include "nongsl.h"

Computer::Computer(Sle& sle): sle(sle), permutation(sle.size), step(0) {
    for (int i = 0; i < sle.size; i++) {
        permutation[i] = i;
    }
}

void Computer::prepare() {
    if (get_a(step)[step] != 0) {
        return;
    }

    for (int line = step + 1; line < sle.size; line++) {
        if (get_a(line)[step] != 0) {
            std::swap(permutation[step], permutation[line]);
            break;
        }
    }
}

void Computer::compute_solution(std::vector<double>& solution) {
    while (step < sle.size) {
        prepare();
        do_step();
        step++;
    }

    for (int i = sle.size - 1; i >= 0; i--) {
        solution[i] = get_b(i);

        double* i_line = get_a(i);
        for (int j = i + 1; j < sle.size; j++) {
            solution[i] -= i_line[j] * solution[j];
        }

        solution[i] /= i_line[i];
    }
}

void Sle::solve(std::vector<double>& solution) {
    Computer computer(*this);

    computer.compute_solution(solution);
}
