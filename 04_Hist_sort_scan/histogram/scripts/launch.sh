#!/bin/bash


EXE_NAME="histogram.out"
EXE=$EXE_NAME

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
# OUTPUT1=../results/$EXE_NAME-$TIME_STAMP
# OUTPUT2=../results/M-$EXE_NAME-$TIME_STAMP
OUTPUT1=../results/
OUTPUT2=../results/C-


for elem_pow in {30..30..2}
do
    for t in {10..10..1}
    do
        elem=$((2**$elem_pow))
        thread=$((2**$t))

        SCRIPT1="compute-sanitizer ../${EXE} --binNum=8192 --Wc=4"
        NAME1=${OUTPUT1}${elem}-${thread}
        sbatch \
            -o $NAME1.txt \
            -p skylake \
            --gres=gpu:rtx_2080_ti:1 \
            --exclusive \
            --wrap="${SCRIPT1}"
    done
done