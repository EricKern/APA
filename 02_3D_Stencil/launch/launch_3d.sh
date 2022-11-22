#!/bin/bash


EXE_NAME="FDTD3d"
EXE_SUFFIX=""
EXE=$EXE_NAME$EXE_SUFFIX

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=../results/$EXE_NAME-$TIME_STAMP

SCRIPT="../FDTD3d/${EXE} --radius="

for i in {1..1..1}
do
NAME=$OUTPUT-$i
sbatch \
    -o 2$NAME.txt \
    -p skylake \
    --gres=gpu:rtx_2080_ti:1 \
    --exclusive \
    --wrap="${SCRIPT}${i}"
done