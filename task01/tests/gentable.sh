taskdir=$(dirname "$0")/..

source "$taskdir/tests/common.sh"

function threads-state {
    local num=$(get-threads-num $1)
    test "" != "$num" && echo -n " + $num threads"
}

function nested-state {
    test true = "$(get-nested-flag $1)" && echo -n " + nested parallelism"
}

function dynamic-state {
    test true = "$(get-dynamic-flag $1)" && echo -n " + dynamic number of threads"
}

function sheduler-state {
    local sf=$(get-shedule-flag $1)
    test "" != "$sf" && echo -n " + $sf sheduling"
}

function sheduler-chunk-state {
    local chunk_size=$(get-shedule-chunk-size $1)

    test "" != "$chunk_size" && echo -n " + chunk size $chunk_size"
}

function get-testee-state {
    local name=$(get-name $1)

    case $name in
        gsl)
            echo -n "GNU Scientific Library"
            ;;
        seq)
            echo -n "Sequential"
            ;;
        simd)
            echo -n "Using simd"
            ;;
        par)
            echo -n "Parallel"
            ;;
        *)
            ;;
    esac

    echo -n "$(threads-state $1)$(dynamic-state $1)$(nested-state $1)$(sheduler-state $1)$(sheduler-chunk-state $1)"
}

function time-from-ns {
    local ns
    let "ns = $1 % 1000"
    local mks
    let "mks = $1 / 1000 % 1000"
    local ms
    let "ms = $1 / 1000000 % 1000"
    local s
    let "s = $1 / 1000000000"
    
    echo -n "$s s $ms ms $mks mks $ns ns"
}

# 1 - result directory

sizes=$(
    for sizefile in "$1"/*; do
        echo "$(basename "$sizefile")"
    done | sort --numeric
)

for size in $sizes; do
    sizefile="$1/$size"
    sorted=$(
        awk '{printf "%s %s %s\n", $2, $1, $3}' < "$sizefile" |
        sort --numeric
    )
    bestpar=$(
        grep -m 1 par <<< "$sorted" |
        awk '{printf "%s", $2}'
    )

    column -s~ -t -R 1 <<< $(
        echo "Size $size:~~|~Time~|~CPU time"
        index=0
        while read result; do
            let "index++"
            testee=$(awk '{print $2}' <<< "$result")
            time=$(awk '{print $1}' <<< "$result")
            cputime=$(awk '{print $3}' <<< "$result")

            echo "$index.~$(get-testee-state "$testee")~|~$(time-from-ns $time)~|~$(time-from-ns $cputime)"
        done <<< $(grep -e "gsl " -e "seq " -e "simd " -e "$bestpar " <<< "$sorted")
    )
    echo
done
