#include <iostream>
#include "test.h"

using std::cin;
using std::cout;

int Test::failedNum = 0;
int Test::totalNum = 0;

void Test::check(bool expr, const char *func, const char  *filename, size_t lineNum) {
    ++totalNum;
    if (!expr) {
        ++failedNum;
        cout << "test failed: " << func << " in " << filename << ":" << lineNum << '\n';
    }
}

void Test::showFinalResult() {
    if (!failedNum)
        cout << "All tests have passed.\n";
    else
        cout << "Failed " << failedNum << " of " << totalNum << " tests.\n";
}

