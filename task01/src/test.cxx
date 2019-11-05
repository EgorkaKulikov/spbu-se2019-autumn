#include <bits/types/struct_timespec.h>
#include <iostream>

#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>

#include "sle.h"

long long get_time() {
    struct timespec timespec;
    clock_gettime(CLOCK_BOOTTIME, &timespec);
    return timespec.tv_sec * 1000000000 + timespec.tv_nsec;
}

long long get_cpu_time() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return rusage.ru_utime.tv_sec * 1000000000 + rusage.ru_utime.tv_usec * 1000;
}

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

    auto begin_time = get_time();
    auto begin_cpu_time = get_cpu_time();

    sle.solve(solution);
    
    auto end_time = get_time();
    auto end_cpu_time = get_cpu_time();

    std::cout << end_time - begin_time << std::endl;
    std::cout << end_cpu_time - begin_cpu_time << std::endl;

    for (auto& v : solution) {
        std::cout << v << std::endl;
    }

    return 0;
}
