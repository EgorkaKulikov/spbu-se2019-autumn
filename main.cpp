#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include "imp.h"

#define EPS 1e-9

using namespace std;


bool compareArr(double* fstArr, double* sndArr, VectorXd third_Arr, int n) {
	for (int i = 0; i < n; i++) {
		if (abs(fstArr[i] - sndArr[i]) > EPS)
			return false;
		if (abs(fstArr[i] - third_Arr(i)) > EPS)
			return false;
		if (abs(third_Arr(i) - sndArr[i]) > EPS)
			return false;
	}
	return true;
}


void experiment_time(double** a, double* b, double* x, 
	MatrixXd eigen_a, VectorXd  eigen_b, int size) {
	ofstream file;
	vector<string> methods = { "consistent", "parallel",  "eigen"};
	VectorXd eigen_res;

	file.open("res.txt", ofstream::app);
	
	if (!file.is_open())
	{
		cout << "Файл не открыт\n";
		exit(1);
	}

	int count_of_experiments = 30;
	int min_time = 1000000;
	int max_time = 0;
	int avg_time = 0;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	file << "Size of matrix:" << size << endl;

	for (const auto method : methods) {
		
		for (int i = 0; i < count_of_experiments; i++) {
			start = std::chrono::system_clock::now();
			if (method == "consistent") {
				if (i == 0)	file << "Consistent time:" << endl;
				consistentImp(a, b, x, size);
			}
			if (method == "parallel") {
				if (i == 0) file << "Parallel time:" << endl;
				parallelImp(a, b, x, size);
			}
			if (method == "eigen") {
				if (i == 0) file << "Eigen time:" << endl;
				eigen_res = eigenImp(eigen_a, eigen_b);
			}
			end = std::chrono::system_clock::now();
			int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
				(end - start).count();

			avg_time += elapsed_seconds;
			max_time = max(max_time, elapsed_seconds);
			min_time = min(min_time, elapsed_seconds);

		}

		file << "Avg time: " << avg_time / count_of_experiments << endl;
		file << "Max time: " << max_time << endl;
		file << "Min time: " << min_time << endl;

		min_time = 1000000;
		max_time = 0;
		avg_time = 0;
	}

	file.close();
}


int main() {

	vector<int> N = {50, 100, 500, 1000};

	for (const auto size : N) {
		double** a = new double* [size];
		double* b = new double[size];
		double* x = new double[size];

		for (int i = 0; i < size; i++) {
			a[i] = new double[size];
		}
		srand(time(0));
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				a[i][j] = rand() % 100;
			}
			b[i] = rand() % 100;
		}
		using namespace Eigen;
		MatrixXd eigen_a(size, size);
		VectorXd  eigen_b(size);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++)
				eigen_a(i, j) = a[i][j];
			eigen_b(i) = b[i];
		}

		if (compareArr(consistentImp(a, b, x, size),
			parallelImp(a, b, x, size), 
			eigenImp(eigen_a, eigen_b),
			size)) {
			cout << "Correct answer" << endl;
		}
		else {
			cout << "Incorrect answer" << endl;
			return 1;
		}
		

#if 0
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				cout << a[i][j] << "\t";
			}
			cout << endl;
		}
#endif // 1


		experiment_time(a, b, x, eigen_a, eigen_b, size);

		for (int i = 0; i < size; i++) {
			delete[] a[i];
		}
		delete[] a;
		delete[] b;
		delete[] x;
	}
	return 0;
}