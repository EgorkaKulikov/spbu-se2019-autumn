#include <cstdio>
#include <cstdlib>

#include "errors.h"

vector<void *> toCleanUp = vector<void *>();
vector<FILE *> toClose = vector<FILE *>();

void finish(errorCode code) {
    for (void *ptr: toCleanUp) {
        free(ptr);
    }

    for (FILE *file: toClose) {
        fclose(file);
    }

    printf(errorMessages[code]);
    exit(code);
}
