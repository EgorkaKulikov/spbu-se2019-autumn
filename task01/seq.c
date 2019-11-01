#include "nongsl.h"

void do_step(struct info* info) {
    double pivot = get_a(info, info->step)[info->step];

    for (int line = info->step + 1; line < info->sle->size; line++) {
        double a = get_a(info, line)[info->step] / pivot;

        for (int column = info->step; column < info->sle->size; column++) {
            get_a(info, line)[column] -= a * get_a(info, info->step)[column];
        }

        *get_b(info, line) -= a * *get_b(info, info->step);
    }
}
