#include "stdio.h"
#include "stdlib.h"

void print_array(int* array, int size)
{
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", array[i]);
    }
}

void generate_random_array(int* arr, int size)
{
    for (int i = 0; i < size; ++i)
    {
        arr[i] = rand() % (size * 10);
    }
}