calc() {
    echo "scale=6;$1" | bc
}

comp() {
    echo "$1 < $2" | bc
}

i_can_wait=160
max_amount_tests=20

g++ -O3 generateMatrix.cpp -o generateMatrix

for optimization in -O0 -O3
do
    g++ -fopenmp $optimization main.cpp -o main

    for size in 125 250 500 1000 2000 4000
    do
        k=0
        s_time=0
        p_time=0
        e_time=0

        while [[ $(comp $k $max_amount_tests) == 1 && $(comp $(calc "$s_time + $p_time + $e_time") $i_can_wait) == 1 ]]
        do
            let "k++"
            ./generateMatrix $size > matrix.txt
            s_time=$(calc "$s_time + $(./main sequential < matrix.txt)")
            p_time=$(calc "$p_time + $(./main parallel < matrix.txt)")
            e_time=$(calc "$e_time + $(./main eigen < matrix.txt)")
            if [[ $(diff -q sx.txt px.txt) != "" || $(diff -q px.txt ex.txt) != "" ]]
            then
                echo "ERROR"
                exit
            fi
        done

        echo "optimization = $optimization"
        echo "matrix size = $size"
        echo "running time:"
        echo "    sequential: $(calc "$s_time / $k")"
        echo "    parallel:   $(calc "$p_time / $k")"
        echo "    eigen:      $(calc "$e_time / $k")"
        echo "-------------------------"

        if [[ $(comp $(calc "($s_time + $p_time + $e_time) / $k * 8") $i_can_wait) == 0 ]]
        then
            break
        fi
    done
done
