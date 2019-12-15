#!/bin/bash

touch gauss_results.txt
> gauss_results.txt
rm gauss
gcc main.c gauss_algos.c -o gauss -lgsl -lgslcblas -fopenmp -O0
exp_N=10
for size_N in {10,100,200,500,1000,2000,4000}
do
  {
    ./gauss "$size_N" "$exp_N" "sequential"
    ./gauss "$size_N" "$exp_N" "gsl"
    ./gauss "$size_N" "$exp_N" "parallel"
    echo ' '
  } >> gauss_results.txt
done
