#!/usr/bin/bash
declare -a SZ=(10 25 50 100 150 200 250 500 750 1000)
declare -a IMP=("sequential" "gsl" "parallel")

if [ -f res.txt ]; then
    rm res.txt
fi

make

for size in "${SZ[@]}"; do
    echo "Matrix size is $size X $size:" >> res.txt
    echo -e >> res.txt
    N=$((size * (size + 1)))
    cat /dev/urandom | tr -dc '0-9' | fold -w 5 | head -n $N > test_"$size".txt
    for imp in "${IMP[@]}"; do
        for ((i = 1; i <= 10; ++i)) do
            valgrind --tool=callgrind --time-stamp=yes --log-file=result.txt ./main "$size" "test_$size.txt" "$imp"
            echo "$(cat result.txt | grep -P -o '(\d{2}:){3}\d{2}.\d{3}' | tail -1 | tail -c 7)" >> result_"$imp".txt
        done
        echo -n "   Average time for $imp implementation is: " >> res.txt
        echo "$(awk '{ t += $1; c++ } END { print t / c }' result_"$imp".txt) seconds" >> res.txt
        rm -f callgrind.out* result*
    done
    echo -e >> res.txt
    diff sequential gsl
    diff gsl parallel
    diff sequential parallel
done

make clean

rm -f callgrind.out* test*txt "${IMP[@]}"
