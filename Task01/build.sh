#!/bin/bash
declare -a methods=("sequential" "gsl" "parallel")

for((i = 10; i < 100; i += 10))
do
    N=$(($i * $i + $i))
    cat /dev/urandom | tr -dc '0-9' | fold -w 3 | head -n $N > test_$i.txt
done

for((i = 100; i <= 1000; i += 100))
do
    N=$(($i * $i + $i))
    cat /dev/urandom | tr -dc '0-9' | fold -w 3 | head -n $N > test_$i.txt
done

for((i = 10; i < 100; i += 10))
do
    echo -e "Matrix size is $i x $i:" >> res.txt
    ./main $i test_$i.txt test
    for method in "${methods[@]}"
    do
        valgrind --tool=callgrind --time-stamp=yes --log-file=result.txt ./main $i test_$i.txt $method
        echo $method: >> res.txt 
        cat result.txt | grep -P -o '(\d{2}:){3}\d{2}.\d{3}' | tail -1 >> res.txt
        rm result.txt
    done
    echo -e >> res.txt
done

for((i = 100; i <= 1000; i += 100))
do
    ./main $i test_$i.txt test
    echo -e "Matrix size is $i x $i:" >> res.txt
    for method in "${methods[@]}"
    do
        valgrind --tool=callgrind --time-stamp=yes --log-file=result.txt ./main $i test_$i.txt $method
        echo $method: >> res.txt 
        cat result.txt | grep -P -o '(\d{2}:){3}\d{2}.\d{3}' | tail -1 >> res.txt
        rm result.txt
    done
    echo -e >> res.txt
done 
