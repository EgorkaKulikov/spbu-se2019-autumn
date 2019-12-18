#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {

  FILE* check;
  
  if (NULL == (check = fopen("out.txt", "r"))) {

    return -1;
  }
  
  int left, right;
  fscanf(check, "%d", &left);

  while (1 == fscanf(check, "%d", &right)) {

    if (left <= right) {

      left = right;
    } else {

      fclose(check);
      fprintf(stderr, "Wrong order after sort.\n");
      return -2;
    }
  }

  fclose(check);
  
  return 0;
}