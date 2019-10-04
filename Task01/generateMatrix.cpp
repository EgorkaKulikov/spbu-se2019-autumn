#include <iostream>

using namespace std;

typedef long double ld;

int main(int argc, char *argv[]) {
    srand(time(0));
    int n = 1;
    ld l = -100, r = 100;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (argc > 2) {
            l = atof(argv[2]);
            if (argc > 3)
                r = atof(argv[3]);
        }
    }
    cout << n << endl;
    for (int i = 0; i < n * (n + 1); i++)
        cout << ld(rand()) * (r - l) / RAND_MAX + l << endl;
}
