main: gauss.o errors.o main.o test.o
	g++ -std=c++11 -O0 -fopenmp -o $@ $^

%.o: %.cpp
	g++ -std=c++11 -c -O0 -fopenmp -o $@ $<

.PHONY = clean

clean:
	rm -f *.o main
