#!/bin/bash

function skip {
    awk 'NR > '$1'{print $0}'
}

echo "$0 $@"

cnt="$1"

make > /dev/null || exit

for size in 5 10 50 100 500 1000; do
    time_gsl=0
    time_seq=0
    time_par=0

    test=$(./gen $size)

    for n in $(seq 1 "$cnt"); do
        res=$(echo "$test" | ./testgsl)
        time=$(echo "$res" | head -1)
        let "time_gsl += time"

        echo "$res" | skip 1 > right_answer

        res=$(echo "$test" | ./testseq)
        time=$(echo "$res" | head -1)
        let "time_seq += time"

        echo "$res" | skip 1 | diff right_answer -

        res=$(echo "$test" | ./testpar)
        time=$(echo "$res" | head -1)
        let "time_par += time"

        echo "$res" | skip 1 | diff right_answer -
    done

    let "time_gsl /= cnt"
    let "time_seq /= cnt"
    let "time_par /= cnt"

    echo "size $size"
    echo "    gsl: $time_gsl"
    echo "    seq: $time_seq"
    echo "    par: $time_par"
done
