#!/bin/bash

gcc main.c linear_solve.c malloc_check.c -o main.o -lgsl -lgslcblas -fopenmp -O0 -Wall