#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char* argv[])
{
  if (2 != argc){
    printf("Incorrect number of arguments. Expected [size]");
    return -1;
  }

  long size = strtol(argv[1], NULL, 10);
  FILE * f;
  
  if ((f = fopen("input.txt", "w")) == NULL){
    return -1;
  }

  srand(time(NULL));

  for (int i = 0; i < size; ++i) {
    fprintf(f, "%d ", rand() % 10000);
  }

  fclose(f);

  return 0;
}