#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "string.h"
#include "bitonic_sorts.cuh"
#include "utils.cuh"

void check_sort(int* array, unsigned long size, char* type)
{
    for (int i=0 ; i < size - 1; i++)
    {
        if (array[i] > array[i + 1]){
            fprintf(stderr, "Array on %s with size %ld sorted incorrectly\n", type, size);
            return;
        }
    }
    printf("Array on %s with size %ld sorted right\n", type, size);
}

void check_content_equals(int* array, int* other, unsigned long size, char* type)
{
    bool elements_less = false;
    bool elements_more = false;
    
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (array[i] == other[j])
            {
                elements_less = true;
            }
            if (array[j] == other[i])
            {
                elements_more = true;
            }
        }
        if (!elements_less)
        {
            printf("Sorted array on %s with size %ld hasn't some elements \
                    which contain in initial array\n", type, size);
            return;
        }
        if (!elements_more)
        {
            printf("Sorted array on %s with size %ld has some elements \
                    which doesn't contain in initial array\n", type, size);
            return;
        }
    }
    printf("Sorted array on %s with size %ld has equal content as initial array\n", type, size);
}

int main()
{
    unsigned long upperBound = 1024 << 10;
    char* typeCPU = "CPU";
    char* typeGPU = "GPU";    

    for (unsigned long size = 1; size <= upperBound; size <<= 1)
    {
        srand(time(NULL));

        size_t size_mem_array = upperBound*sizeof(int);
        int* array = (int*) malloc(size_mem_array);
        int* temp_array = (int*) malloc(size_mem_array);

        generate_random_array(array, size);
            
        memcpy(temp_array, array, size_mem_array);

        bitonic_sort(temp_array, size);

        check_sort(temp_array, size, typeCPU);
        check_content_equals(array, temp_array, size, typeCPU);
            
        memcpy(temp_array, array, size * sizeof(int));

        bitonic_sort_gpu(temp_array, size);

        check_sort(temp_array, size, typeGPU);
        check_content_equals(array, temp_array, size, typeGPU);
        
        free(array);
        free(temp_array);
        printf("\n");
    }

    return 0;
}