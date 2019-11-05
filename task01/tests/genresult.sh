taskdir=$(dirname "$0")/..

source "$taskdir/tests/common.sh"

# 1 - destination directory
# 2 - stats directory

test $# = 2 || exit 1
test -e "$1" && { rmdir "$1" || exit 1; }
mkdir --parents "$1"

for sizedir in "$2"/*; do
    sizefile="$1/$(basename "$sizedir")"

    for testeedir in "$sizedir"/*; do
        testee=$(basename "$testeedir")
        testeetime=0
        testeecputime=0
        testscnt=0

        for statfile in "$testeedir"/*; do
            echo "$statfile"
            testtime=0
            testcputime=0
            runscnt=0

            while read result; do
                runtime=$(awk '{print $1}' <<< "$result")
                runcputime=$(awk '{print $2}' <<< "$result")

                let "runscnt++"
                let "testtime += runtime"
                let "testcputime += runcputime"
            done < "$statfile"

            let "testscnt++"
            let "testeetime += testtime / runscnt"
            let "testeecputime += testcputime / runscnt"
        done

        let "testeetime /= testscnt"
        let "testeecputime /= testscnt"

        echo "$testee $testeetime $testeecputime" >> "$sizefile"
    done
done
