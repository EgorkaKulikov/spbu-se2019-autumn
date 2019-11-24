#include "nongsl.h"

#include <omp.h>

void Computer::do_step() {
    double* step_line = get_a(step);
    double pivot = step_line[step];

    for (int line = step + 1; line < sle.size; line++) {
        double* cur_line = get_a(line);
        double a = cur_line[step] / pivot;

        #pragma omp for simd
        for (int column = step; column < sle.size; column++) {
            cur_line[column] -= a * step_line[column];
        }

        get_b(line) -= a * get_b(step);
    }
}
