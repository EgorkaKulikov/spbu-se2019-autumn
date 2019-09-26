#!/bin/bash

gfortran ./Task.f95 -o par.run -fopenmp

for i in `seq 1 3`; do

    ./par.run < ./test/$i.in > ./test/$i.out
    
    cmp ./test/$i.test ./test/$i.out

    if [ $? -eq 0 ]

        then echo "Test $i passed."

        else echo "Test $i failed.\n."
    fi

done

rm ./test/*.out