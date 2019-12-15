#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

int inf = INT_MAX;

void swap(int *a, int *b){
     int c = *a;
     *a = *b;
     *b = c;
}

void bitSeqSort(int* l, int* r, short inv) {
	if (r - l <= 1) return;
	int *m = l + (r - l) / 2;
	for (int *i = l, *j = m; i < m && j < r; i++, j++) {
		if (inv ^ (*i > *j)) swap(i, j);
	}
	bitSeqSort(l, m, inv);
	bitSeqSort(m, r, inv);
}
void makeBitonic(int* l, int* r) {
	if (r - l <= 1) return;
	int *m = l + (r - l) / 2;
	makeBitonic(l, m);
	bitSeqSort(l, m, 0);
	makeBitonic(m, r);
	bitSeqSort(m, r, 1);
}
void bitonicSort(int* l, int* r) {
	int* a;
	a = (int*)malloc((r-l) * sizeof(int));
	int current = 0;
	for (int *i = l; i < r; i++)
		a[current++] = *i;
	makeBitonic(a, a + (r-l));
	bitSeqSort(a, a + (r-l), 0);
	current = 0;
	for (int *i = l; i < r; i++)
		*i = a[current++];
	free(a);
}

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Invalid number of arguments\n");
		return 1;
	}
	
	int n = 1, i = 0;

	FILE* input = fopen(argv[1], "r");
	if (input == 0) {
		printf("Error in opening file\n");
		return 2;
	}

	int sz;
	if (fscanf(input, "%d", &sz) != 1) {
		printf("Error while reading file 1\n");
		return 3;
	}

	int* nums;
	while (n < sz) n *= 2;
	nums = (int*)malloc(n * sizeof(int));

	for (i = 0; i < sz; i++) {
		if (fscanf(input, "%d", nums + i) != 1) {
			printf("Error while reading file 2\n");
			return 3;
		}
	}	

	fclose(input);
	i = sz;
	while (i < n) nums[i++] = inf;
	
	clock_t startTime = clock();
	bitonicSort(nums, nums + n);
	clock_t endTime = clock();
	double totalTime = (double) (endTime - startTime) / CLOCKS_PER_SEC / 20;
	
	//test
	short test = 1;
	for(i = 1; i<sz; i++){
		if(nums[i]<nums[i-1]) test = 0;
	}
	if(test == 0){
		printf("Array is not sorted\n");
		return 4;
	}
	
	printf("Time CPU: %f\n", totalTime);
	free(nums);
	return 0;
}
