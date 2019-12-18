#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_sort.h"

int main(int argc, char** argv) {

  if (2 != argc) {

    return -1;
  }

  FILE* input;

  if (NULL == (input = fopen(argv[1], "r"))) {

    return -1;
  }

  int* array = (int*) malloc(1e7 * sizeof(int));
  int i = 0;

  while (1 == fscanf(input, "%d", &array[i])) {

    ++i;
  }

  fclose(input);

  gpu_sort(array, i);
  FILE* output;
  
  if (NULL == (output = fopen("out.txt", "w"))) {

    return -1;
  }

  for (int j = 0; j < i; ++j) {

    fprintf(output, "%d ", array[j]);
  }

  fprintf(output, "\n");

  fclose(output);

  free(array);

  return 0;
}