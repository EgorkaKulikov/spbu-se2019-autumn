#include <cstdio>

#include "test.h"
#include "errors.h"

using namespace std;

int main() {
    char filename[] = {'*', '.', 't', 'x', 't', 0};
    for (int i = 0; i < 9; i++) {
        filename[0] = (char)i + '0';
        test(filename);
    }

    finish(noErrorCode);
}
