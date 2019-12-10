#include <stdlib.h>
#include "imp_sequential.h"

double *sequential(double *a, double *b, int size)
{
	double *x = malloc(sizeof(*x) * size);
                           
	// Прямой ход -- O(n^3)
	for (int k = 0; k < size - 1; k++)
	{
		// Исключение x_i из строк k+1...n-1
		double pivot = a[k * size + k];	
		for (int i = k + 1; i < size; i++)
		{
			// Из уравнения (строки) i вычитается уравнение k
			double lik = a[i * size + k] / pivot;
			for (int j = k; j < size; j++)
				a[i * size + j] -= lik * a[k * size + j];
			b[i] -= lik * b[k];
		}
	}
	// Обратный ход -- O(n^2)
	for (int k = size - 1; k >= 0; k--)
	{
		x[k] = b[k];
		for (int i = k + 1; i < size; i++)
			x[k] -= a[k * size + i] * x[i];
		x[k] /= a[k * size + k];
	}
	return x;
}