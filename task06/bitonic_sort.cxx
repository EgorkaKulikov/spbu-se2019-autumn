
#include <vector>

#include "utils.hxx"

bool bitonic_sort(std::vector<int>& data) {
	int number_of_stages = get_number_of_stages(data);

	for (int stage = 1; stage <= number_of_stages; stage++) {
		int block_size = 1 << stage;
		int number_of_blocks = data.size() >> stage;
		int remain_size = data.size() & (block_size - 1);

		for (int pass = 0; pass < stage; pass++) {
			int step = block_size >> 1;

			for (int block = 0; block < number_of_blocks; block++) {
				for (int i = 0; i < step; i++) {
					bitonic_swap(data.data(), i, block, block_size, pass);
				}
			}

			if (remain_size != 0) {
				for (int i = 0; i + step < remain_size; i++) {
					bitonic_swap(data.data(), i, number_of_blocks, block_size, pass);
				}
			}

			block_size >>= 1;
			number_of_blocks <<= 1;
		}
	}

	return true;
}
