#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include "test.h"
#include "solution.h"

int tests_run = 0;

static char* test_seq_gsl(int n, double *a, double *b) {

    double *x1 = malloc(sizeof(*x1) * n); 
    double *x2 = malloc(sizeof(*x2) * n); 

    x1 = sequential(n, a, b);
    x2 = gsl(n, a, b);

    for (int i = 0; i < n; i++)
        mu_assert("results are different seq gsl", (fabs(x1[i] - x2[i]) < 0.0001));

    free(x1);
    free(x2);
    
    return 0;
}

static char* test_seq_par(int n, double *a, double *b) {

    double *x1 = malloc(sizeof(*x1) * n); 
    double *x2 = malloc(sizeof(*x2) * n); 

    x1 = sequential(n, a, b);
    x2 = parallel(n, a, b);

    for (int i = 0; i < n; i++) 
        mu_assert("results are different seq par", (fabs(x1[i] - x2[i]) < 0.0001));

    free(x1);
    free(x2);

    return 0;
}

static char * all_tests(int n, double *a, double *b) {
    
     mu_run_test(test_seq_gsl);
     mu_run_test(test_seq_par);

     return 0;
 }
 
int run_tests(int n, double *a, double *b) {

     char *result = all_tests(n, a, b);

     if (result != 0)
         printf("%s\n", result);
     else 
         printf("ALL TESTS PASSED\n");

     printf("Tests run: %d\n", tests_run);
 
     return result != 0;
 }
