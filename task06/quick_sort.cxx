
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

//static void parallel_sort(int* array, int begin, int end, int depth, int max_depth)
//{
//    if (depth > max_depth) {
//        sequential_sort(array, begin, end);
//        return;
//    }
//
//    int pivot = partition(array, begin, end);
//    auto thread1 = std::thread(parallel_sort, array, begin, pivot - 1, depth + 1, max_depth);
//    auto thread2 = std::thread(parallel_sort, array, pivot + 1, end, depth + 1, max_depth);
//    thread1.join();
//    thread2.join();
//}

bool quick_sort(std::vector<int>& data) {
    //parallel_sort(array, 0, size - 1, 0, get_max_depth());
    sequential_sort(data.data(), 0, data.size() - 1);
	return true;
}
