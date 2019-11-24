void sequential_solve(size_t size
                      , double *matrix[size]
                      , double *b_vector
                      , double *x_vector);

void omp_solve(size_t size
                    , double *matrix[size]
                    , double *b_vector
                    , double *x_vector);

void gsl_solve(size_t size
               , double matrix[size * size]
               , double *b_vector
               , double *x_vector);

void generate_vector(size_t size, double *b_vector);

void generate_matrix(size_t size, double *matrix[size]);

void copy_sle(int size
              , double *matrix[size]
              , double *b_vector
              , double *matrix_copy[size]
              , double *b_vector_copy);