#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int size;

    if (argc != 2) {
        exit(1);
    }

    size = strtol(argv[1], NULL, 0);

    printf("%d\n", size);

    srand(time(NULL));
    for (int i = 0; i < size * size + size; i++) {
        printf("%lf\n", 1.0 * rand());
    }

    return 0;
}
