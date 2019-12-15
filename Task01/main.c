#include <stdio.h>
#include "gauss_algos.h"
#include <time.h>

double *generate_matrix(int n) {
    double *a = malloc(sizeof(*a) * n * n);
    for (int i = 0; i < n; i++) { // Инициализация
        // srand(i * (n + 1));
        for (int j = 0; j < n; j++)
            a[i * n + j] = rand() % 100 + 1;
    }
    return a;
}

double *generate_column(int n) {
    double *b = malloc(sizeof(*b) * n);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 100 + 1;
    }
    return b;
}

int main(int argc, char *argv[]) {

    char need_cmp = 0;
    if (argc < 4) {
        exit(1);
    }
    int n = atoi(argv[1]);
    int exp_n = atoi(argv[2]);
    char *algo_name = argv[3];

    clock_t start, end;
    double clock_sum;
    double *(*algo)(int, double *, double *);
    switch (algo_name[0]) {
        case 's' : {
            algo = &seq_gauss;
            need_cmp = 1;
            break;
        }
        case 'g' : {
            algo = &GSL_gauss;
            break;
        }
        case 'p' : {
            algo = &parallel_gauss;
            need_cmp = 1;
            break;
        }
    }
    clock_sum = 0;
    for (int j = 0; j < exp_n; j++) {
        double *a = generate_matrix(n);
        double *b = generate_column(n);
        start = clock();
        double *x = algo(n, a, b);
        end = clock();
        clock_sum += ((double)(end - start)) / CLOCKS_PER_SEC;

        if (need_cmp) {
            double *gsl_x = GSL_gauss(n, a, b);
            // Сравнение векторов
            for (int i = 0; i < n; i++) {
                if (fabs(x[i] - gsl_x[i]) > 0.0001) {
                    fprintf(stderr, "Invalid result: elem %d: %f %f\n", i, x[i], gsl_x[i]);
                    break;
                }
            }
            free(gsl_x);
        }
        free(b);
        free(a);
        free(x);
    }
    printf("Gaussian Elimination (%s): n %d, mean-time (sec) %.6f\n", algo_name, n, ((double) clock_sum) / (double)exp_n);
    return 0;
}
