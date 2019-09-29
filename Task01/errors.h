#ifndef ERRORS
#define ERRORS

#include <vector>
#include <cstdio>

using namespace std;

const char *const errorMessages[] = {
      ""
    , "Error: file not found!\n"
    , "Error: incorrect input!\n"
    , "Error: insufficient memory!\n"
    , "Error: invalid arguments!\n"
};

extern vector<void *> toCleanUp;
extern vector<FILE *> toClose;

enum errorCode {noErrorCode, fileErrorCode, inputErrorCode, memoryErrorCode, argumentErrorCode};

void finish(errorCode code);

#endif //ERRORS
