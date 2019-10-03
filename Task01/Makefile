all: par seq

par: par.c
	g++ -openmp par.c -o par

seq: seq.c
	g++ seq.c -o seq

clean: 
	rm -rf *.o par seq
