CFLAGS= -Wall -O2 -Werror -I "include"

all: bin main

bin:
	mkdir -p bin

main: bin/main.o bin/solution.o  bin/test.o
	gcc $(CFLAGS) -fopenmp -o main bin/main.o bin/solution.o bin/test.o -lgsl -lgslcblas -lm

bin/main.o: src/main.c include/solution.h include/test.h
	gcc $(CFLAGS) -c -fopenmp src/main.c -o bin/main.o -lgsl -lgslcblas -lm

bin/solution.o: src/solution.c include/solution.h
	gcc $(CFLAGS) -c -fopenmp src/solution.c -o bin/solution.o -lgsl -lgslcblas -lm 

bin/test.o: src/test.c include/test.h include/solution.h
	gcc $(CFLAGS) -c -fopenmp src/test.c -o bin/test.o -lgsl -lgslcblas -lm
	
clean:
	rm -rf bin main
