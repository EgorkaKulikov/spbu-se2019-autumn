void solve_seq(int size, double **matrix,
               double *RHS, double *solutions);

void solve_par(int size, double **matrix,
               double *RHS, double *solutions);

void solve_gsl(int size, double *matrix,
               double *RHS, double *solutions);

void generate_matrix(int size, double **matrix, double *RHS);
