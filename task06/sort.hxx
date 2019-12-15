#pragma once

typedef bool (*sort_t)(std::vector<int>& data);

extern bool quick_sort(std::vector<int>& data);
extern bool bitonic_sort(std::vector<int>& data);
extern bool gpu_bitonic_sort(std::vector<int>& data);
