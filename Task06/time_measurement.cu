#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "string.h"
#include "bitonic_sorts.cuh"
#include "utils.cuh"

#define AMOUNT_MEASUREMENTS 20

int main(char* args)
{
    clock_t start;
    unsigned long upperBound = 1024 << 13;
    
    for (unsigned size = 1024; size <= upperBound; size <<= 1)
    {        
        srand(time(NULL));
        
        size_t size_mem_array = size*sizeof(int);

        int* array = (int*) malloc(size_mem_array);
        int* temp_array = (int*) malloc(size_mem_array);
        
        double timeGPU = 0;
        double timeCPU = 0;

        generate_random_array(array, size);
        
        for (int i = 0; i < 20; i++)
        {
            memcpy(temp_array, array, size_mem_array);
            
            start = clock();
            bitonic_sort(temp_array, size);
            timeCPU += ((double) (clock() - start));

            memcpy(temp_array, array, size_mem_array);

            start = clock();
            bitonic_sort_gpu(temp_array, size);
            timeGPU += ((double) (clock() - start));
        }

        double avg_timeCPU = timeCPU  / CLOCKS_PER_SEC / AMOUNT_MEASUREMENTS;
        double avg_timeGPU = timeGPU  / CLOCKS_PER_SEC / AMOUNT_MEASUREMENTS;

        printf("%ld %f %f\n", size, avg_timeCPU, avg_timeGPU);
        
        free(array);
        free(temp_array);
    }

    return 0;
}