#pragma once

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

inline CUDA_DEVICE void swap(int& a, int& b) {
	int tmp = a;
	a = b;
	b = tmp;
}

inline CUDA_DEVICE void bitonic_swap(int* data, int index, int block_index, int block_size, int pass) {
	int step = block_size >> 1;
	int i = block_index * block_size + index;

	bool ascending = ((block_index >> pass) & 1) == 0;

	if (ascending && data[i] > data[i + step] || !ascending && data[i] < data[i + step]) {
		swap(data[i], data[i + step]);
	}
}

inline int get_number_of_stages(const std::vector<int>& data) {
	int result = 0;
	int additional_step = 0;

	for (int i = data.size(); i > 1; i >>= 1) {
		if (i & 1) {
			additional_step = 1;
		}
		result++;
	}

	return result + additional_step;
}
