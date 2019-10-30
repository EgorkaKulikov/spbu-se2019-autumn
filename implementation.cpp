#include "imp.h"

double* consistentImp(double** a, double* b, double* x, int n) {
	// Ïðÿìîé õîä -- O(n^3) 
	for (int k = 0; k < n - 1; k++) {
		// Èñêëþ÷åíèå x_i èç ñòðîê k+1...n-1 
		double pivot = a[k][k];
		for (int i = k + 1; i < n; i++) {
			// Èç óðàâíåíèÿ (ñòðîêè) i âû÷èòàåòñÿ óðàâíåíèå k 
			double lik = a[i][k] / pivot;
			for (int j = k; j < n; j++)
				a[i][j] -= lik * a[k][j];
			b[i] -= lik * b[k];
		}
	}
	// Îáðàòíûé õîä -- O(n^2)
	for (int k = n - 1; k >= 0; k--) {
		x[k] = b[k];
		for (int i = k + 1; i < n; i++)
			x[k] -= a[k][i] * x[i];
		x[k] /= a[k][k];
	}
#if 0
	for (int i = 0; i < n; i++) {
		std::cout << x[i] << " ";
	}
	std::cout << std::endl;
#endif
	return x;
}

VectorXd eigenImp(MatrixXd a, VectorXd b) {
	return a.householderQr().solve(b);
}


double* parallelImp(double** a, double* b, double* x, int n) {
	for (int k = 0; k < n - 1; k++) {
		double pivot = a[k][k];
		for (int i = k + 1; i < n; i++) {
			double lik = a[i][k] / pivot;
			#pragma omp simd
			for (int j = k; j < n; j++) {
				a[i][j] -= lik * a[k][j];
			}
			b[i] -= lik * b[k];
		}
	}

	for (int k = n - 1; k >= 0; k--) {
		x[k] = b[k];
		#pragma omp simd
		for (int i = k + 1; i < n; i++) {
			x[k] -= a[k][i] * x[i];
		}
		x[k] /= a[k][k];
	}
#if 0
	for (int i = 0; i < n; i++) {
		std::cout << x[i] << " ";
	}
	std::cout << std::endl;
#endif
	return x;

 }
