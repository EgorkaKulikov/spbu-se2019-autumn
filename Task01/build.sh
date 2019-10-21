#!/bin/bash

gcc main.c linear_solve.c -o main.o -lgsl -lgslcblas -fopenmp -O0