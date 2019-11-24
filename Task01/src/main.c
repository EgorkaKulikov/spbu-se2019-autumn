#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "test.h"
#include "solution.h"

int main(int argc, char *argv[]) {

    int n = atoi(argv[1]);
    
    double *a = malloc(sizeof(*a) * n * n);
    double *b = malloc(sizeof(*b) * n); 
    
    const char* file_name = (const char*)argv[2];
    const char* type = (const char*)argv[3];

    FILE* input_file = fopen(file_name, "r");

    for (int i = 0; i < n; i++) { 
        for (int j = 0; j < n; j++)
            fscanf(input_file, "%le", &a[i * n + j]);
        fscanf(input_file, "%le",  &b[i]);
    }

    fclose(input_file);

    if (strcmp(type, "sequential") == 0) {
       double *x_seq = sequential(n, a, b);
       free(x_seq);
    } else if (strcmp(type, "gsl") == 0) {
       double *x_gsl = gsl(n, a, b);
       free(x_gsl);
    } else if (strcmp(type, "parallel") == 0) {
       double *x_par = parallel(n, a, b);
       free(x_par);
    } else if (strcmp(type, "test") == 0) {
        run_tests(n, a, b);
    } else 
        printf("Invalid third argument");

    free(b);
    free(a);

    return 0;
}
