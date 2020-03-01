#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int random() {
    return (rand() % 2001) - 1000;
}

int main(int argc, char** argv) {
    srand(clock());
    int arr_size = atoi(argv[1]);
    FILE *fout = fopen(argv[2], "w");
    fprintf(fout, "%d", arr_size);
    for (int i = 0; i < arr_size; i++) {
        fprintf(fout, " %d", random());
    }
    fclose(fout);
}
