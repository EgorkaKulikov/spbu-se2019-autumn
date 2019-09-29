#!/bin/bash

gfortran ./Task.f95 -o par.run -fopenmp

./par.run < ./test/1.in > ./test/1.out
    
cmp ./test/1.test ./test/1.out

if [ $? -eq 0 ]

    then echo "Test passed."

    else echo "Test failed.\n."
fi

done

rm ./test/*.out