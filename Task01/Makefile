CXX = g++
CXXFLAGS = -Wall -Werror -pedantic -std=c++11 -Iinclude -c

all: bin main

gauss_test: bin bin/gauss.o bin/test.o bin/gausstest.o bin/tests.o
	$(CXX) bin/gauss.o bin/test.o bin/gausstest.o bin/tests.o -o gauss_test -O0 -lgsl -lgslcblas -lm -fopenmp

main: bin/main.o bin/gauss.o
	$(CXX) bin/main.o bin/gauss.o -o main -O0 -lgsl -lgslcblas -lm -fopenmp

bin/tests.o: test/gausstest.cpp include/gausstest.h test/test.cpp include/test.h test/tests.cpp
	$(CXX) $(CXXFLAGS) test/tests.cpp -o bin/tests.o -O0 -lgsl -lgslcblas -lm -fopenmp

bin/test.o: test/test.cpp include/test.h
	$(CXX) $(CXXFLAGS) test/test.cpp -o bin/test.o -O0 -lgsl -lgslcblas -lm -fopenmp

bin/gausstest.o: test/gausstest.cpp include/gausstest.h test/test.cpp include/test.h
	$(CXX) $(CXXFLAGS) test/gausstest.cpp -o bin/gausstest.o -O0 -lgsl -lgslcblas -lm -fopenmp

bin/main.o: src/main.cpp include/gauss.h
	$(CXX) $(CXXFLAGS) src/main.cpp -o bin/main.o -O0 -lgsl -lgslcblas -lm -fopenmp

bin/gauss.o: src/gauss.cpp include/gauss.h
	$(CXX) $(CXXFLAGS)  src/gauss.cpp -o bin/gauss.o -O0 -lgsl -lgslcblas -lm -fopenmp

clean:
	rm -rf bin main gauss_test

bin:
	mkdir -p bin
