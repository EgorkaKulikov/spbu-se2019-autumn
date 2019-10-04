#include <vector>
#include <omp.h>

using namespace std;

typedef long double ld;

vector<ld> parallelGauss(vector<vector<ld>> matrix_Ab_T) {
    omp_set_num_threads(omp_get_max_threads());
    int n = matrix_Ab_T.size() - 1;
    for (int num = 0; num < n; num++) {
        for (int i = num + 1; i <= n; i++) {
            matrix_Ab_T[i][num] /= matrix_Ab_T[num][num];
            #pragma omp simd
            for (int j = num + 1; j < n; j++)
                matrix_Ab_T[i][j] -= matrix_Ab_T[num][j] * matrix_Ab_T[i][num];
        }
    }
    for (int num = n - 1; num >= 0; num--) {
        #pragma omp simd
        for (int i = 0; i < num; i++)
            matrix_Ab_T[n][i] -= matrix_Ab_T[num][i] * matrix_Ab_T[n][num];
    }
    return matrix_Ab_T[n];
}
