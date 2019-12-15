
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <ctime>
#include <cstdio>

#include "sort.hxx"

int is_sorted(const std::vector<int>& data) {
	int cnt = 0;

	for (int i = 1; i < data.size(); i++) {
		if (data[i - 1] > data[i]) {
			cnt++;
		}
	}

	return cnt;
}

void fill_random(std::vector<int>& data, int begin) {
	for (auto& elem : data) {
		elem = std::rand();
	}
}

struct sort_info {
	sort_t sort;
	int time;
	std::string name;

	sort_info(sort_t sort, std::string name) : sort(sort), name(name) {}
};

#define SORT(name) sort_info ( \
	name##_sort,               \
	#name                      \
)

static int multiplier = 2;

int main() {
	srand(time(NULL));

	std::vector<struct sort_info> sorts;
	sorts.push_back(SORT(quick));
	sorts.push_back(SORT(bitonic));
	sorts.push_back(SORT(gpu_bitonic));

	std::vector<int> data;
	bool no_error = true;

	data.resize(1 << 10);
	fill_random(data, 0);

	while (no_error) {
		std::cerr << data.size() << std::endl;

		for (auto& sort_info : sorts) {
			std::vector<int> temp_data(data);
			
			int begin_time = std::clock();
			no_error = sort_info.sort(temp_data);
			int end_time = std::clock();

			if (!no_error) {
				std::cerr << sort_info.name << ": Error during sorting" << std::endl;
				return 1;
			}

			int count = is_sorted(temp_data);

			if (count != 0) {
				std::cerr << sort_info.name << ": Invalid sorting " << std::to_string(count) << std::endl;
				//return 1;
			}

			sort_info.time = end_time - begin_time;
		}

		std::cout << data.size() << std::endl;

		for (auto& sort_info : sorts) {
			std::cout << sort_info.name << '~' << sort_info.time << std::endl;
		}

		try {
			int old_size = data.size();
			int new_size = old_size * multiplier;
			data.resize(new_size);
			fill_random(data, old_size);
		} catch (...) {
			std::cerr << "Can not increase array" << std::endl;
			return 1;
		}
	}

	return 0;
}
