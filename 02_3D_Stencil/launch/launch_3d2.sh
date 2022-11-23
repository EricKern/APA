#!/bin/bash


EXE_NAME="FDTD3d"
EXE_SUFFIX=""
EXE=$EXE_NAME$EXE_SUFFIX

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=../results/2-$EXE_NAME-$TIME_STAMP

# SCRIPT="compute-sanitizer --destroy-on-device-error kernel ../FDTD3d/${EXE} --radius="
SCRIPT="../FDTD3d/${EXE} --radius="

for i in {4..4..1}
do
NAME=$OUTPUT-$i
sbatch \
    -o ${NAME}.txt \
    -p skylake \
    --gres=gpu:rtx_2080_ti:1 \
    --exclusive \
    --wrap="${SCRIPT}${i} --kernel2"
done