#include <sstream>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2) {
        exit(1);
    }

    int size;
    std::stringstream(argv[1]) >> size;

    std::cout << size << std::endl;

    std::default_random_engine generator {std::random_device()()};
    std::uniform_real_distribution<double> distribution(-10, 10);

    for (int i = 0; i < size * size + size; i++) {
        std::cout << distribution(generator) << std::endl;
    }

    return 0;
}
