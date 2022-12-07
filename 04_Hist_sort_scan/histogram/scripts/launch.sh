#!/bin/bash


EXE_NAME="histogram.out"
EXE=$EXE_NAME

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
# OUTPUT1=../results/$EXE_NAME-$TIME_STAMP
# OUTPUT2=../results/M-$EXE_NAME-$TIME_STAMP
OUTPUT1=../results/


for Wc_pow in {0..4..1}
do
    for binNum_pow in {8..13..1}
    do
        binNum=$((2**$binNum_pow))
        Wc=$((2**$Wc_pow))

        # build wrap script for sbatch
        SCRIPT1="../${EXE} --binNum=$binNum --Wc=$Wc"
        # build output file name
        padded_Wc=`printf %02d $Wc`
        padded_binNum=`printf %04d $binNum`
        OUT_NAME1=${OUTPUT1}Wc${padded_Wc}-binNum$padded_binNum
        sbatch \
            -o $OUT_NAME1.txt \
            -p skylake \
            --gres=gpu:rtx_2080_ti:1 \
            --exclusive \
            --wrap="${SCRIPT1}"
    done
done