taskdir=$(dirname "$0")/..

source "$taskdir/tests/common.sh"

# 1 - destination directory
# 2 - sizes
# 3 - tests per size

test -e "$1" && { rmdir "$1" || exit 1; }
mkdir --parents "$1"

iterate_seq=$(seq 1 $3)

for size in $(get-sizes "$2"); do
    sizedir="$1/$size"
    mkdir "$sizedir"

    for n in $iterate_seq; do
        testdir="$sizedir/test_$n"
        mkdir "$testdir"
        echo "Generate $testdir"
        "$taskdir/bin/gen" $size > "$testdir/test"
        "$taskdir/bin/solve" < "$testdir/test" > "$testdir/answer"
    done    
done
