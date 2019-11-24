#!/bin/bash

gcc solve.c test.c -fopenmp -lgsl -lgslcblas -o gauss.exe
./gauss.exe > output.txt