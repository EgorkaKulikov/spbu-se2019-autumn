#!/bin/bash

taskdir=$(dirname "$0")/..

source "$taskdir/tests/common.sh"

# 1 - destination directory
# 2 - tests directory
# 3 - runs per test

test -e "$1" && { rmdir "$1" || exit 1; }
mkdir --parents "$1"

iterate_seq=$(seq 1 $3)

for sizedir in "$2"/*; do
    size=$(basename "$sizedir")
    dstsizedir="$1/$size"
    mkdir "$dstsizedir"

    for testee in $(gen-all-testees $size); do
        testeedir="$dstsizedir/$testee"
        mkdir "$testeedir"

        name=$(get-name $testee)

        if test par = $name; then
            export OMP_NUM_THREADS=$(get-threads-num $testee)
            export OMP_DYNAMIC=$(get-dynamic-flag $testee)
            export OMP_NESTET=$(get-nested-flag $testee)
            export OMP_SHEDULE=$(get-shedule-flag $testee)
    
            chunk_size=$(get-shedule-chunk-size $testee)
    
            if test "" != "$chunk_size"; then
                export OMP_SHEDULE="$OMP_SHEDULE",$chunk_size
            fi
        fi

        for testdir in "$sizedir"/*; do
            statfile="$testeedir/$(basename "$testdir")"

            for n in $iterate_seq; do
                res=$("$taskdir/bin/test$name" < "$testdir/test")
                time=$(head -2 <<< "$res")
                is_ok=$(skip 2 <<< "$res" | diff "$testdir/answer" - 2>&1 > /dev/null && echo yes || echo no)

                echo $time $is_ok >> "$statfile"
                echo "$testdir $testee $n" $time "$is_ok"
            done
        done
    done
done
