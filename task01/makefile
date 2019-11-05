executables = bin/gen bin/testseq bin/testgsl bin/testpar bin/testsimd bin/solve

all: $(executables)

CXX=g++
CXXFLAGS=-O3 -Wall -Wextra -pedantic -Werror

omp_objects=build/simd.o build/par.o
other_objects=build/test.o build/nongsl.o build/gsl.o build/gen.o build/seq.o build/solve.o
objects=$(other_objects) $(omp_objects)

$(executables): bin
$(objects): build

bin/gen: build/gen.o
bin/solve: build/solve.o build/gsl.o
bin/testgsl: build/test.o build/gsl.o
bin/testseq: build/test.o build/nongsl.o build/seq.o
bin/testpar: build/test.o build/nongsl.o build/par.o
bin/testsimd: build/test.o build/nongsl.o build/simd.o

build/gen.o: src/gen.cxx
build/solve.o: src/solve.cxx
build/test.o: src/test.cxx
build/simd.o: src/simd.cxx
build/par.o: src/par.cxx
build/seq.o: src/seq.cxx
build/gsl.o: src/gsl.cxx
build/nongsl.o: src/nongsl.cxx
build/simd.o build/par.o build/seq.o build/nongsl.o: src/nongsl.h

bin/testpar_flags = -fopenmp
bin/testsimd_flags = -fopenmp
bin/solve_flags = -lgsl -lgslcblas
bin/testgsl_flags = -lgsl -lgslcblas

$(executables):
	$(CXX) -o $@ $(filter %.o,$^) $($@_flags)

build/simd.o_flags = -fopenmp
build/par.o_flags = -fopenmp

$(objects): src/sle.h
	$(CXX) -c $(CXXFLAGS) -o $@ $(filter %.cxx,$^) $($@_flags)

build:
	mkdir build

bin:
	mkdir bin

clean:
	rm -rf build bin
