#!/bin/bash


EXE_NAME="threadFenceReduction"
EXE=$EXE_NAME

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
# OUTPUT1=../results/$EXE_NAME-$TIME_STAMP
# OUTPUT2=../results/M-$EXE_NAME-$TIME_STAMP
OUTPUT1=../results/
OUTPUT2=../results/M-


for elem_pow in {6..26..2}
do
    for t in {7..10..1}
    do
        elem=$((2**$elem_pow))
        thread=$((2**$t))

        SCRIPT1="../${EXE} --n=$elem --threads=$thread"
        SCRIPT2="../${EXE} --n=$elem --threads=$thread --multipass"
        NAME1=${OUTPUT1}${elem}-${thread}
        NAME2=${OUTPUT2}${elem}-${thread}
        sbatch \
            -o $NAME1.txt \
            -p skylake \
            --gres=gpu:rtx_2080_ti:1 \
            --exclusive \
            --wrap="${SCRIPT1}"

        sbatch \
            -o $NAME2.txt \
            -p skylake \
            --gres=gpu:rtx_2080_ti:1 \
            --exclusive \
            --wrap="${SCRIPT2}"
    done
done