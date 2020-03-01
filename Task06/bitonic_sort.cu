#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_AMOUNT_GPU_THREADS 1024

void bitonic_sort_CPU_compare(int *arr,
                              int arr_size,
                              int log2_block_size,
                              bool second_half_of_block_is_reversed,
                              int threadIdx,
                              int blockIdx,
                              int blockDim) {
    int thread_index = threadIdx + blockDim * blockIdx,
        block_size = 1 << log2_block_size,
        log2_amount_threads_in_block = log2_block_size - 1,
        amount_threads_in_block = block_size >> 1,
        block_index = thread_index >> log2_amount_threads_in_block,
        thread_index_in_block = thread_index & (amount_threads_in_block - 1),
        first_index,
        second_index;
    if (second_half_of_block_is_reversed) {
        first_index = amount_threads_in_block - thread_index_in_block - 1 + block_index * block_size;
        second_index = first_index + thread_index_in_block * 2 + 1;
    }
    else {
        first_index = thread_index_in_block + block_index * block_size;
        second_index = first_index + amount_threads_in_block;
    }
    if (second_index < arr_size && arr[first_index] > arr[second_index]) {
        int helper = arr[first_index];
        arr[first_index] = arr[second_index];
        arr[second_index] = helper;
    }
}

int *bitonic_sort_CPU(int *arr, int arr_size) {
    int *local_arr;
    local_arr = (int*) malloc(sizeof(int) * arr_size);
    memcpy(local_arr, arr, sizeof(int) * arr_size);

    int amount_gpu_blocks = (arr_size - 1) / MAX_AMOUNT_GPU_THREADS + 1,
        total_amount_gpu_threads = (arr_size + 1) >> 1,
        amount_gpu_threads_in_block = (total_amount_gpu_threads - 1) / amount_gpu_blocks + 1;

    for (int stage = 0; (1 << stage) < arr_size; stage++) {
        for (int log2_block_size = stage + 1; log2_block_size > 0; log2_block_size--) {
            bool second_half_of_block_is_reversed = (log2_block_size == stage + 1);
            for (int blockIdx = 0; blockIdx < amount_gpu_blocks; blockIdx++) {
                for (int threadIdx = 0; threadIdx < amount_gpu_threads_in_block; threadIdx++) {
                    bitonic_sort_CPU_compare(
                        local_arr,
                        arr_size,
                        log2_block_size,
                        second_half_of_block_is_reversed,
                        threadIdx,
                        blockIdx,
                        amount_gpu_threads_in_block
                    );
                }
            }
        }
    }
    return local_arr;
}

__global__ void bitonic_sort_GPU_compare(int *arr,
                                         int arr_size,
                                         int log2_block_size,
                                         bool second_half_of_block_is_reversed) {
    int thread_index = threadIdx.x + blockDim.x * blockIdx.x,
        block_size = 1 << log2_block_size,
        log2_amount_threads_in_block = log2_block_size - 1,
        amount_threads_in_block = block_size >> 1,
        block_index = thread_index >> log2_amount_threads_in_block,
        thread_index_in_block = thread_index & (amount_threads_in_block - 1),
        first_index,
        second_index;
    if (second_half_of_block_is_reversed) {
        first_index = amount_threads_in_block - thread_index_in_block - 1 + block_index * block_size;
        second_index = first_index + thread_index_in_block * 2 + 1;
    }
    else {
        first_index = thread_index_in_block + block_index * block_size;
        second_index = first_index + amount_threads_in_block;
    }
    if (second_index < arr_size && arr[first_index] > arr[second_index]) {
        int helper = arr[first_index];
        arr[first_index] = arr[second_index];
        arr[second_index] = helper;
    }
}

int *bitonic_sort_GPU(int *arr, int arr_size) {
    int *local_arr;
    cudaMalloc(&local_arr, sizeof(int) * arr_size);
    cudaMemcpy(local_arr, arr, sizeof(int) * arr_size, cudaMemcpyHostToDevice);

    int amount_gpu_blocks = (arr_size - 1) / MAX_AMOUNT_GPU_THREADS + 1,
        total_amount_gpu_threads = (arr_size + 1) >> 1,
        amount_gpu_threads_in_block = (total_amount_gpu_threads - 1) / amount_gpu_blocks + 1;

    for (int stage = 0; (1 << stage) < arr_size; stage++) {
        for (int log2_block_size = stage + 1; log2_block_size > 0; log2_block_size--) {
            bool second_half_of_block_is_reversed = (log2_block_size == stage + 1);
            bitonic_sort_GPU_compare<<<amount_gpu_blocks, amount_gpu_threads_in_block>>>(
                local_arr,
                arr_size,
                log2_block_size,
                second_half_of_block_is_reversed
            );
        }
    }

    int *res_arr;
    res_arr = (int*) malloc(sizeof(int) * arr_size);

    cudaMemcpy(res_arr, local_arr, sizeof(int) * arr_size, cudaMemcpyDeviceToHost);

    cudaFree(local_arr);

    return res_arr;
}

int main(int argc, char **argv) {
    bool input_file_is_declared = false,
         output_file_is_declared = false,
         take_sorting_time = false;
    FILE *input_file = 0,
         *output_file = 0;
    bool processing_unit_declared_as_central = true;
    for (int argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "-in") == 0 || strcmp(argv[argi], "--input_file") == 0) {
            if (argi + 1 == argc) {
                return 1;
            }
            argi++;
            input_file_is_declared = true;
            input_file = fopen(argv[argi], "r");
        }
        else if (strcmp(argv[argi], "-out") == 0 || strcmp(argv[argi], "--output_file") == 0) {
            if (argi + 1 == argc) {
                return 1;
            }
            argi++;
            output_file_is_declared = true;
            output_file = fopen(argv[argi], "w");
        }
        else if (strcmp(argv[argi], "-c") == 0 || strcmp(argv[argi], "--on-cpu") == 0) {
            processing_unit_declared_as_central = true;
        }
        else if (strcmp(argv[argi], "-g") == 0 || strcmp(argv[argi], "--on-gpu") == 0) {
            processing_unit_declared_as_central = false;
        }
        else if (strcmp(argv[argi], "-tt") == 0 || strcmp(argv[argi], "--take-time") == 0) {
            take_sorting_time = true;
        }
        else {
            printf("%s is unknown option.", argv[argi]);
            return 2;
        }
    }

    int arr_size, *arr;
    if (input_file_is_declared) {
        fscanf(input_file, "%d", &arr_size);
        arr = (int*) malloc(arr_size * sizeof(int));
        for (int i = 0; i < arr_size; i++) {
            fscanf(input_file, "%d", arr + i);
        }
        fclose(input_file);
    }
    else {
        scanf("%d", &arr_size);
        arr = (int*) malloc(arr_size * sizeof(int));
        for (int i = 0; i < arr_size; i++) {
            scanf("%d", arr + i);
        }
    }

    int *sorted_arr;
    clock_t start_time = clock();
    if (processing_unit_declared_as_central) {
        sorted_arr = bitonic_sort_CPU(arr, arr_size);
    }
    else {
        sorted_arr = bitonic_sort_GPU(arr, arr_size);
    }
    clock_t stop_time = clock();

    if (output_file_is_declared) {
        fprintf(output_file, "%d\n", arr_size);
        for (int i = 0; i < arr_size; i++) {
            fprintf(output_file, "%d\n", sorted_arr[i]);
        }
        fclose(output_file);
    }
    else {
        printf("%d\n", arr_size);
        for (int i = 0; i < arr_size; i++) {
            printf("%d\n", sorted_arr[i]);
        }
    }

    if (take_sorting_time) {
        if ( ! output_file_is_declared)
            printf("\n");
        printf("Sorting time = %d ms.", stop_time - start_time);
    }

    free(sorted_arr);

    return 0;
}
