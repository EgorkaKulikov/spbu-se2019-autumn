CFLAGS= -Wall -Werror -Wpedantic -O0  -fopenmp -lgsl -lgslcblas -lm

main: sequential.o gsl.o parallel.o main.o
	gcc $(CFLAGS) -o $@ $^

%.o: %.c
	gcc $(CFLAGS) -c -o $@ $< 

.PHONY = clean

clean :
	rm -f *.o main