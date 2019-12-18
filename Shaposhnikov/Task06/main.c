#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "bitonic_sort.h"

void check_array(int *array, int size) {
  for (int i = 1; i < size; i++) {
    if (array[i-1] > array[i]) {
      printf("Mistake at positions %d and %d numbers are %d and %d\n", (i-1), i, array[i-1], array[i]);
    }
  }
}

int main(int argc, char* argv[])
{
  if (2 != argc){
    printf("Incorrect number of arguments. Expected [file name]");
    return -1;
  }

  FILE* file;
  if ((file = fopen(argv[1], "r")) == NULL){
    printf("Inccorect input. Ensure [%s] exists", argv[1]);
    return -1;
  }

  //1e7 * 2 seems to be a limit of array and it allows to sort up to 2^24 ints
  int* gpu_array = (int*) malloc(1e7 * 2 * sizeof(int));
  int* cpu_array = (int*) malloc(1e7 * 2 * sizeof(int));
  int size = 0;

  while (1 == fscanf(file, "%d", &gpu_array[size])) {
    cpu_array[size] = gpu_array[size];
    ++size;
  }

  fclose(file);

  clock_t gpu_start, gpu_stop, cpu_start, cpu_stop;

  gpu_start = clock();
  gpu_bitonic_sort(gpu_array, size);
  gpu_stop = clock();

  check_array(gpu_array, size);
  free(gpu_array);

  cpu_start = clock();
  bitonic_sort(cpu_array, size);
  cpu_stop = clock();

  check_array(cpu_array, size);
  free(cpu_array);

  float g_time = ((double) (gpu_stop - gpu_start)) / CLOCKS_PER_SEC;
  float c_time = ((double) (cpu_stop - cpu_start)) / CLOCKS_PER_SEC;

  FILE* output = fopen("output.txt", "a+");
  fprintf(output, "%d %f %f\n", size, g_time, c_time);

  printf("  Gpu time is:\n  _%f\n", g_time);
  printf("  Cpu time is:\n  _%f\n", c_time);
}