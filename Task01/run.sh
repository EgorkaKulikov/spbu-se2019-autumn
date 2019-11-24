#!/bin/bash

gcc main.c algorithms.c -o main -lgsl -lgslcblas -lm -O2 -fopenmp

for size in 25 50 100 500 1000 2000
do
    ./main $size 20
done
