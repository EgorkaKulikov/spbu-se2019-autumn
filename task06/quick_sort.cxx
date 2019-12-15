
#include <vector>

#include "utils.hxx"

static int get_max_depth() {
    return 2;
}

static int partition(int* array, int begin, int end)
{
    int x = array[begin];
    swap(array[begin], array[end]);

    int current = begin;
    for (int i = begin; i < end; ++i)
    {
        if (array[i] < x) {
            swap(array[current], array[i]);
            current++;
        }
    }

    swap(array[current], array[end]);

    return current;
}

static void sequential_sort(int* array, int begin, int end)
{
    if (end < begin) return;
    int pivot = partition(array, begin, end);
    sequential_sort(array, begin, pivot - 1);
    sequential_sort(array, pivot + 1, end);
}

bool quick_sort(std::vector<int>& data) {
    sequential_sort(data.data(), 0, data.size() - 1);
	return true;
}
