
#include <stdlib.h>
#include <stdio.h>

void *malloc_check(size_t n)
{
    //Checking for NULL malloc return value
    void *memPtr = malloc(n);
    if (memPtr == NULL)
    {
        fprintf(stderr, "Unable to allocate memory!\n");
        exit(4);
    }
    return memPtr;
}