function get-info {
    awk -F_ '{print $'$1'}' <<< $2
}

function get-name {
    get-info 1 $1
}

function get-threads-num {
    get-info 2 $1
}

function get-dynamic-flag {
    get-info 3 $1
}

function get-nested-flag {
    get-info 4 $1
}

function get-shedule-flag {
    get-info 5 $1
}

function get-shedule-chunk-size {
    get-info 6 $1
}

function gen-par-testees { #1 - size
    for i in $(seq 2 4); do
        for df in true false; do
            for nf in true false; do
                local base=par_${i}_${df}_${nf}

                for sf in static dynamic guided; do
                    local div=$i

                    echo "${base}_${sf}_${1}"

                    while test $div -le $1; do
                        echo ${base}_${sf}_$(($1 / $div))
                        let "div *= i"
                    done

                    echo ${base}_${sf}
                done

                echo ${base}_auto
            done
        done
    done
}

function gen-all-testees { # 1 - size
    echo -n "gsl seq simd $(gen-par-testees $1)"
}

function get-sizes { # 1 - sizes
    awk -F, '{
        for (i = 1; i < NF; i++)
            printf "%s ", $i
        printf "%s", $NF
    }' <<< "$1"
}

function skip {
    awk 'NR > '$1'{print $0}'
}
