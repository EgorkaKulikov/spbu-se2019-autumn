#include <string.h>

#include "nongsl.h"

void prepare(struct info* info) {
    if (get_a(info, info->step)[info->step] != 0) {
        return;
    }

    for (int line = info->step + 1; line < info->sle->size; line++) {
        if (get_a(info, line)[info->step] != 0) {
            int temp = info->permutation[info->step];
            info->permutation[info->step] = info->permutation[line];
            info->permutation[line] = temp;
            break;
        }
    }
}

void compute_solution(struct info* info, double* solution) {
    for (int i = info->sle->size - 1; i >= 0; i--) {
        solution[i] = *get_b(info, i);

        for (int j = i + 1; j < info->sle->size; j++) {
            solution[i] -= get_a(info, i)[j] * solution[j];
        }

        solution[i] /= get_a(info, i)[i];
    }
}

void solve(struct sle* sle, double* solution) {
    int permutation[sle->size];

    for ( int i = 0; i < sle->size; i++ ) {
        permutation[i] = i;
    }

    struct info info = {
                        .sle = sle,
                        .permutation = permutation,
                        .step = 0,
    };

    while (info.step < sle->size) {
        prepare(&info);
        do_step(&info);
        info.step++;
    }

    compute_solution(&info, solution);
}
