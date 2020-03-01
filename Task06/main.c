#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    FILE *fout_cpu, *fout_gpu;
    int depth_requests_size = 5;
    int step_size = 100000000;
    int* requests_size = (int*) malloc(sizeof(int) * (1 << depth_requests_size));
    requests_size[0] = step_size;
    for (int i = 0, j = 1, ind = 1; i < depth_requests_size; i++, step_size >>= 1, j <<= 1) {
        for (int _size = (step_size >> 1), k = 0; k < j; _size += step_size, k++, ind++) {
            requests_size[ind] = _size;
        }
    }
    for (int i = -1; i < (1 << depth_requests_size); i++) {
        printf("\nGenerating array of size %d elements", requests_size[abs(i)]);
        char command[100];
        sprintf(command, "generate_array.exe %d input.txt", requests_size[abs(i)]);
        system(command);

        printf("\nSorting on CPU\n");
        system("bitonic_sort.exe -in input.txt -out output_cpu.txt -c -tt");

        printf("\nSorting on GPU\n");
        system("bitonic_sort.exe -in input.txt -out output_gpu.txt -g -tt");

        printf("\nChecking answers\n");
        fout_cpu = fopen("output_cpu.txt", "r");
        fout_gpu = fopen("output_gpu.txt", "r");

        int arr_size_cpu, arr_size_gpu,
            pred_elem_arr_cpu, pred_elem_arr_gpu,
            elem_arr_cpu, elem_arr_gpu;

        fscanf(fout_cpu, "%d", &arr_size_cpu);
        fscanf(fout_gpu, "%d", &arr_size_gpu);

        if (arr_size_cpu != arr_size_gpu) {
            printf("Error: Size of arrays are not equal");
            return -1;
        }

        if (arr_size_cpu == 0) continue;

        fscanf(fout_cpu, "%d", &pred_elem_arr_cpu);
        fscanf(fout_gpu, "%d", &pred_elem_arr_gpu);

        if (pred_elem_arr_cpu != pred_elem_arr_gpu) {
            printf("Error: Elements with index 0 in arrays are not equal");
            return -1;
        }

        for (int j = 1; j < arr_size_cpu; j++) {
            fscanf(fout_cpu, "%d", &elem_arr_cpu);
            fscanf(fout_gpu, "%d", &elem_arr_gpu);

            if (elem_arr_cpu < pred_elem_arr_cpu) {
                printf("Error: CPU's array is not sorted");
                return -1;
            }

            if (elem_arr_gpu < pred_elem_arr_gpu) {
                printf("Error: GPU's array is not sorted");
                return -1;
            }

            if (elem_arr_cpu != elem_arr_gpu) {
                printf("Error: Elements with index %d in arrays are not equal", j);
                return -1;
            }

            pred_elem_arr_cpu = elem_arr_cpu;
            pred_elem_arr_gpu = elem_arr_gpu;
        }

        fclose(fout_cpu);
        fclose(fout_gpu);
    }
    return 0;
}
