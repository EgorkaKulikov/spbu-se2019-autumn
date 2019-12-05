#!/bin/bash

touch result.txt

gcc gsl.c parallel_sle.c sequential.c main.c -o gaussian -lgsl -lgslcblas -fopenmp -O0 -w -Wpedantic

for	size in {10,100,200,500,1000,2000,3000}
do
	./gaussian "$size" >> result 
done