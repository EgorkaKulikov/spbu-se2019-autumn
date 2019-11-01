EXE=gen testgsl testseq testpar

CFLAGS=-O2 -Wall -Wextra -pedantic -Werror -std=c11

all: $(EXE)

gen: gen.o
	cc -o $@ $^

testgsl: test.o gsl.o
	cc -o $@ $^ -lgsl -lgslcblas

testseq: test.o nongsl.o seq.o
	cc -o $@ $^

testpar: test.o nongsl.o par.o
	cc -o $@ $^ -fopenmp

par.o: par.c
	cc -c $(CFLAGS) $^ -fopenmp

clean:
	rm -rf $(EXE) *.o
