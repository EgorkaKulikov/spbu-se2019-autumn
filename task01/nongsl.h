#pragma once

#include "sle.h"

struct info {
    struct sle* sle;
    int* permutation;
    int step;
};

extern void do_step(struct info* info);

inline double* get_a(struct info* info, int line) {
    return info->sle->a + info->permutation[line] * info->sle->size;
}

inline double* get_b(struct info* info, int line) {
    return info->sle->b + info->permutation[line];
}
