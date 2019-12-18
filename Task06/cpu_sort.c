#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void quicksort(int* array, int left, int right) {

  if (left == right) {
      
    return;
  }

  int pivot = array[left + rand() % (right - left)];
  int i = left, j = right, buf;

  while (i < j) {

    while (array[i] < pivot && i < right) {

      ++i;
    }

    while (array[j] > pivot && j > left) {

      --j;
    }

    if (i <= j) {

      buf      = array[j];
      array[j] = array[i];
      array[i] = buf;
      
      ++i;
      --j;
    }
  }

  if (j > left) {

    quicksort(array, left, j);
  }

  if (i < right) {

    quicksort(array, i, right);
  }
}

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

  quicksort(array, 0, i - 1);
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