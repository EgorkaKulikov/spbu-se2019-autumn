#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<stdbool.h>

#include "solutions.h"

#define max_error 1e-5
#define experiment_amount 30

double **matrix;
double *RHS;

bool verify_equality(double *solution1, double *solution2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (solution1[i] - solution2[i] > max_error)
            return false;
    }
    return true;
}

void get_methods_time(double *seq_time, double *par_time,
                      double *gsl_time, int matrix_size)
{
    for (int i = 0; i < experiment_amount; i++)
    {
         for (int method = 0; method < 3; method++)
         {
             clock_t start, end;
             double **matrix_copy = malloc(matrix_size*sizeof(double *));
             for (int j = 0; j < matrix_size; j++)
             {
                 matrix_copy[j] = malloc(matrix_size*sizeof(double));
                 memcpy(matrix_copy[j], matrix[j], matrix_size*sizeof(double));
             }
             double *RHS_copy = malloc(matrix_size*sizeof(double));
             memcpy(RHS_copy, RHS, matrix_size*sizeof(double));
             double *solution = malloc(matrix_size*sizeof(double));
             switch (i)
             {
             case 0:
                start = clock();
                solve_seq(matrix_size, matrix_copy, RHS_copy, solution);
                end = clock();
                *seq_time += (double)(end - start) / CLOCKS_PER_SEC;
                break;
             case 1:
                start = clock();
                solve_par(matrix_size, matrix_copy, RHS_copy, solution);
                end = clock();
                *par_time += (double)(end - start) / CLOCKS_PER_SEC;
                break;
             case 2:
                start = clock();
                solve_gsl(matrix_size, matrix_copy, RHS_copy, solution);
                end = clock();
                *gsl_time += (double)(end - start) / CLOCKS_PER_SEC;
                break;
             default:
                break;
             }
         }
    }
    *seq_time /= (double)experiment_amount;
    *par_time /= (double)experiment_amount;
    *gsl_time /= (double)experiment_amount;

}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
      printf("expected 2 arguments");
      exit(1);
  }

  int matrix_size = atoi (argv[1]);
  matrix = malloc(matrix_size*sizeof(double *));
  if (matrix == NULL)
  {
      printf("Failed to allocate memory");
      exit(2);
  }

  for (int i = 0; i < matrix_size; i++)
  {
      matrix[i] = malloc(matrix_size*sizeof(double));
      if (matrix[i] == NULL)
      {
          printf("Failed to allocate memory");
          exit(2);
      }
  }

  // right-hand side
  RHS = malloc(matrix_size*sizeof(double));
  if (RHS == NULL)
  {
      printf("Failed to allocate memory");
      exit(2);
  }

  double *solution_seq = malloc(matrix_size*sizeof(double));
  double *solution_par = malloc(matrix_size*sizeof(double));
  double *solution_gsl = malloc(matrix_size*sizeof(double));
  if (solution_seq==NULL || solution_par==NULL || solution_gsl==NULL)
  {
      printf("Failed to allocate memory");
      exit(2);
  }

  generate_matrix(matrix_size, matrix, RHS);

  double seq_time = 0, par_time = 0, gsl_time = 0;

  get_methods_time(&seq_time, &par_time, &gsl_time, matrix_size);


  double **matrix_copy_seq = malloc(matrix_size*sizeof(double *));
  double **matrix_copy_par = malloc(matrix_size*sizeof(double *));
  double *matrix_copy_gsl = malloc(matrix_size*matrix_size*sizeof(double));
  for (int j = 0; j < matrix_size; j++)
  {
      matrix_copy_seq[j] = malloc(matrix_size*sizeof(double));
      matrix_copy_par[j] = malloc(matrix_size*sizeof(double));
      memcpy(matrix_copy_seq[j], matrix[j], matrix_size*sizeof(double));
      memcpy(matrix_copy_par[j], matrix[j], matrix_size*sizeof(double));
  }

  for (int i = 0; i < matrix_size; i++)
    for (int j = 0; j < matrix_size; j++)
        matrix_copy_gsl[i*matrix_size + j] = matrix[i][j];

  double *RHS_copy_seq = malloc(matrix_size*sizeof(double));
  double *RHS_copy_par = malloc(matrix_size*sizeof(double));
  double *RHS_copy_gsl = malloc(matrix_size*sizeof(double));
  memcpy(RHS_copy_seq, RHS, matrix_size*sizeof(double));
  memcpy(RHS_copy_par, RHS, matrix_size*sizeof(double));
  memcpy(RHS_copy_gsl, RHS, matrix_size*sizeof(double));

  solve_seq(matrix_size, matrix_copy_seq, RHS_copy_seq, solution_seq);
  solve_par(matrix_size, matrix_copy_par, RHS_copy_par, solution_par);
  solve_gsl(matrix_size, matrix_copy_gsl, RHS_copy_gsl, solution_gsl);
  if (verify_equality(solution_seq, solution_gsl, matrix_size) &&
      verify_equality(solution_par, solution_gsl, matrix_size))
        printf ("all is correct\n");
  else
    printf ("mistake in solution\n");

  printf("results for matrix size %d:\n", matrix_size);
  printf("sequential solution:%f seconds\n", seq_time);
  printf("parallel solution:%f seconds\n", par_time);
  printf("gsl solution:%f seconds\n", gsl_time);

  for (int i=0; i<matrix_size; i++)
  {
      free(matrix[i]);
  }
  free(matrix);
  free(RHS);
  free(solution_seq);
  free(solution_par);
  free(solution_gsl);

  return 0;
}
