#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void swap(int *arr, int i, int j) {
	int temp;
	temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
}

void bitonic_sort(int *arr, unsigned int log_len) {
	int arr_len = 1 << log_len;

	for (int seq_size = 2; seq_size <= arr_len; seq_size <<= 1) {
		for (int comp_dist = seq_size >> 1; comp_dist > 0; comp_dist >>= 1) {
			for (int item = 0; item < arr_len; item++) {
				int pair_item = item | comp_dist;
				bool in_first_half = (item & seq_size) == 0;

				//First half of the sequence is sorted ascending, second - descending
				if (in_first_half && arr[item] > arr[pair_item]
					|| !in_first_half && arr[item] < arr[pair_item])
					swap(arr, item, pair_item);
			}
		}
	}
}