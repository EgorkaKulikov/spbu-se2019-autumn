#include <gsl/gsl_linalg.h>
#include <stdio.h>
#include "string.h"
#include <time.h>
#include "algorithms.h"

#define NUM_WAYS 3
#define EPSILON 1e9

int num_replies;

double* matrix;
double* const_vector;
double* result_seq_vector;
double* result_gsl_vector;
double* result_par_vector;
double* result_time;

enum way_decision
{
    SEQ_IMP,            //sequential implementation without gsl facilities
    GSL_IMP,            //gsl implementation
    PARALLEL            //parallel implementation
};

void allocate_mem_for_equation(int matrix_size)
{
    matrix = malloc(sizeof(double) * matrix_size * matrix_size); 
    const_vector = malloc(sizeof(double) * matrix_size); // vector b in Ax=b 

    result_seq_vector = malloc(sizeof(double) * matrix_size);
    result_par_vector = malloc(sizeof(double) * matrix_size);
    result_gsl_vector = malloc(sizeof(double) * matrix_size);
    result_time = malloc(sizeof(double) * NUM_WAYS);

    if (matrix == NULL || const_vector == NULL
        || result_seq_vector == NULL || result_gsl_vector == NULL
        || result_par_vector == NULL
        || result_time == NULL)
    {
        printf("Error: cannot allocate memory for equation");
        exit(2);
    }
}


void init_equation(int matrix_size)
{
    for (int line = 0; line < matrix_size; line++) {
        for (int column = 0; column < matrix_size; column++)
            matrix[line * matrix_size + column] = rand() % 100 + 1;
        const_vector[line] = rand() % 100 + 1;
    }
}

void solve_equation(int matrix_size)
{
    time_t start, end;
    
    double* buf_matrix = malloc(sizeof(double) * matrix_size * matrix_size);
    double* buf_const_vector = malloc(sizeof(double) * matrix_size);

    if (buf_matrix == NULL || buf_const_vector == NULL)
    {
        printf("Error: cannot allocate memory for matrix or/and const_vector buffers");
        exit(2);
    }
    
    for (int method = SEQ_IMP; method <= PARALLEL; method++)
    {
        int sum_squares_results = 0;
        result_time[method] = 0;
        
        for (int num_exp = 0; num_exp < num_replies; num_exp++) {
            memcpy(buf_matrix, matrix, sizeof(double) * matrix_size * matrix_size);
            memcpy(buf_const_vector, const_vector, sizeof(double) * matrix_size);

            start = clock();
            switch(method) 
            {
                case SEQ_IMP:
                    imp_seq(buf_matrix, buf_const_vector, result_seq_vector, matrix_size);
                    break;
                case GSL_IMP:
                    imp_gsl(buf_matrix, buf_const_vector, result_gsl_vector, matrix_size);
                    break;
                case PARALLEL:
                    imp_par(buf_matrix, buf_const_vector, result_par_vector, matrix_size);
                    break;
            }
            end = clock();

            result_time[method] += (double) (end - start) / CLOCKS_PER_SEC;
        }

        result_time[method] /= num_replies;
    }

    free(buf_matrix);
    free(buf_const_vector);
}

int check_results(int matrix_size)
{
    for (int i = 0; i < matrix_size; ++i)
    {
        if (abs(result_gsl_vector[i] - result_par_vector[i]) > EPSILON 
            || abs(result_seq_vector[i] != result_par_vector[i]) > EPSILON)
        {
            return -1;
        }
    }

    return 0;
}

void write_results(int eq_vectors_flag, int matrix_size)
{
    FILE *output_file;
    if ((output_file = fopen("results.txt", "a+")) == NULL) {
        printf("Error: results.txt file cannot be open");
        exit(2);
    }
    
    if (eq_vectors_flag == -1)
    {
        fprintf(output_file, "Matrix size: %d \n \
                              Results are not correct\n", matrix_size);
        for (int i =0; i< matrix_size; ++i)
        {
            fprintf(output_file, "%f \t", result_seq_vector[i]);
            fprintf(output_file, "%f \t", result_gsl_vector[i]);
            fprintf(output_file, "%f \n", result_par_vector[i]);
        }
    } else
    {
        fprintf(output_file, "Matrix size %d \n \
                          -sequential solution\t %f \n \
                          -gsl solution\t %f \n \
                          -parallel solution\t %f \n\n"
                         , matrix_size
                         , result_time[SEQ_IMP]
                         , result_time[GSL_IMP]
                         , result_time[PARALLEL]);
    }

    fclose(output_file);
}

void free_global_variables()
{
    free(matrix);
    free(const_vector);
    free(result_seq_vector);
    free(result_par_vector);
    free(result_time);
}

int main(int argc, char *argv[]) {
    
    if (argc != 3)
    {
        printf("Error: not enough arguements");
        exit(1);
    }
    
    int matrix_size = atoi(argv[1]);
    num_replies = atoi(argv[2]);  

    allocate_mem_for_equation(matrix_size);

    srand(time(NULL));
    init_equation(matrix_size);

    solve_equation(matrix_size);

    write_results(check_results(matrix_size), matrix_size);
    
    free_global_variables();

    return 0;
}