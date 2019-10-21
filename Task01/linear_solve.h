void sequential_solve(int size
                      , double *matrix[size]
                      , double *b
                      , double *x);

void omp_solve(int size
                    , double *matrix[size]
                    , double *b
                    , double *x);

void gsl_solve(int size
               , double matrix[size * size]
               , double *b
               , double *x);

void generate_vector(int size, double *b);

void generate_matrix(int size, double *matrix[size]);

void copy_sle(int size
              , double *matrix[size]
              , double *b
              , double *matrix_copy[size]
              , double *b_copy);