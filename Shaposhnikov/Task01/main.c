#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#include "imp_sequential.h"
#include "imp_gsl.h"
#include "imp_parallel.h"

#define EXPERIMENTS_AMOUNT 20
#define EPS 1e-4

double* init_matrix(int size, double* b) {
    double *a = malloc(sizeof(*a) * size * size);
	for (int i = 0; i < size; i++)
	{
		srand(i * (size+ 1));
		for (int j = 0; j < size; j++)
			a[i * size + j] = rand() % 100 + 1;

		b[i] = rand() % 100 + 1;
	}
	return a;
}

int main(int argc, char *argv[]) {
    if (argc != 2)
	{
		fprintf(stderr, "Incorrect number of arguments. Expected: [size]");
		exit(1);
	}

	int matrix_size = atoi(argv[1]);
	if (matrix_size <= 0) 
	{
		fprintf(stderr, "[size] should be more than 0, now is %d", matrix_size);
		exit(1);
	}

	double gsl_total_time = 0;
	double seq_total_time = 0;
	double parallel_total_time = 0;

	for (int i = 0; i < EXPERIMENTS_AMOUNT; i++)
	{
		double *vector_b = malloc(sizeof(*vector_b) * matrix_size);
		double *matrix = init_matrix(matrix_size, vector_b);

		//both ways as i don't know which one's correct
		double *matrix_gsl = malloc(sizeof(double*) * matrix_size * matrix_size);
		double *vector_gsl = malloc(sizeof(*vector_gsl) * matrix_size);
		double *matrix_seq = malloc(sizeof(double*) * matrix_size * matrix_size);
		double *vector_seq = malloc(sizeof(*vector_seq) * matrix_size);

		//making 3 sets of data for 3 algorithms
		memcpy(matrix_gsl, matrix, matrix_size * matrix_size * sizeof( double*));
		memcpy(matrix_seq, matrix, matrix_size * matrix_size * sizeof( double*));
		memcpy(vector_gsl, vector_b, matrix_size * sizeof( double*));
		memcpy(vector_seq, vector_b, matrix_size * sizeof( double*));
		
		clock_t start_gsl = clock();
		double* gsl_res = gsl_implementation(matrix_gsl, vector_gsl, matrix_size);
		clock_t end_gsl = clock();
		gsl_total_time += ((double) end_gsl - (double) start_gsl)/ CLOCKS_PER_SEC;

		clock_t start_seq = clock();
		double* seq_res = sequential(matrix_seq, vector_seq, matrix_size);
		clock_t end_seq = clock();
		seq_total_time += ((double) end_seq - (double) start_seq)/ CLOCKS_PER_SEC;

		clock_t start_parallel = clock();
		double* parallel_res = parallel_sle(matrix, vector_b, matrix_size);
		clock_t end_parallel = clock();
		parallel_total_time += ((double) end_parallel - (double) start_parallel)/ CLOCKS_PER_SEC;

		free(matrix);
		free(vector_b);
		free(matrix_gsl);
		free(vector_gsl);
		free(matrix_seq);
		free(vector_seq);

		for (int k = 0; k < matrix_size; k++)
		{
			if (2*parallel_res[k] - gsl_res[k] - seq_res[k] > EPS)
			{
				fprintf(stderr, "Diverging results in different algorithms");
				exit(3);
			}
		}
	}

	printf("For [size] = %d x %d approximate time:\n", matrix_size, matrix_size);
	printf("	Gsl implementation: %f \n", gsl_total_time / EXPERIMENTS_AMOUNT);
	printf("	Sequential implementation: %f \n", seq_total_time / EXPERIMENTS_AMOUNT);
	printf("	Parallel implementation: %f \n\n", parallel_total_time / EXPERIMENTS_AMOUNT);
	return 0;
}