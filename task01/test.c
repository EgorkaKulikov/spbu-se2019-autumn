#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include "sle.h"

static long long get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

int main(void) {
    int size;

    scanf("%d", &size);

    double data[size * size + size];

    for ( int i = 0; i < size * size + size; i++ ) {
        scanf( "%lf", &data[i] );
    }

    struct sle sle = {
                      .a = data,
                      .b = data + size * size,
                      .size = size,
    };

    double solution[size];

    long long begin_time = get_time();
    solve(&sle, solution);
    long long end_time = get_time();

    printf( "%lld\n", end_time - begin_time );
    for ( int i = 0; i < size; i++ ) {
        printf( "%.4lf\n", solution[i] );
    }

    return 0;
}
