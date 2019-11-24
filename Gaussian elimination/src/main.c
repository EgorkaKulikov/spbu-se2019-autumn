#include <stdio.h>
#include <stdlib.h>
#include <string.h>

 #include "sequential.h"
 #include "gsl.h"
 #include "parallel.h"

int main(int argc, char *argv[]) {
    
    int matrix_size = atoi(argv[1]);
    const char* file_name = (const char*)argv[2];
    char* imp_type = (char*)argv[3]; 

    FILE* input_file = fopen(file_name, "r");

    if (NULL == input_file) {
        fprintf(stderr, "Can't open file %s", file_name);
        exit(1);
    }

    double *matrix = malloc(sizeof(*matrix) * matrix_size * matrix_size);
    double *equations = malloc(sizeof(*equations) * matrix_size);
    double *solution = malloc(sizeof(*solution) * matrix_size);

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            fscanf(input_file, "%lf", &matrix[i * matrix_size + j]);
        }
        fscanf(input_file, "%lf", &equations[i]);
    }

    #if 0
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            printf("%12.4f ", matrix[i * matrix_size + j]);
        }
        printf(" | %12.4f\n", equations[i]);
    }
    #endif

    if (0 == strcmp(imp_type, "sequential")) {
        sequential_imp(matrix, equations, solution, matrix_size);
    } else if (0 == strcmp(imp_type, "gsl")) {
        gsl_imp(matrix, equations, solution, matrix_size);
    } else if (0 == strcmp(imp_type, "parallel")) {
        parallel_imp(matrix, equations, solution, matrix_size);
    }

    FILE* output_file = fopen(imp_type, "w");

    if (NULL == output_file) {
        fprintf(stderr, "Can't open file %s", imp_type);
        exit(1);
    }

    for (int i = 0; i < matrix_size; ++i) {
        fprintf(output_file, "%12.4f\n", solution[i]);
    }
    
    fclose(input_file);
    fclose(output_file);

    free(solution);
    free(equations);
    free(matrix);
    
    return 0;
}