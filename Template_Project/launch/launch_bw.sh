#!/bin/bash


EXE_NAME="bandwidth"
EXE_SUFFIX=".out"
EXE=$EXE_NAME$EXE_SUFFIX

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=../results/$EXE_NAME-$TIME_STAMP

SCRIPT="../build/${EXE}.out --devices 0 --csv ${OUTPUT}.csv"

sbatch \
    -o $OUTPUT.txt \
    -p skylake \
    --gres=gpu:rtx_2080_ti:1 \
    --exclusive \
    --wrap="${SCRIPT}"