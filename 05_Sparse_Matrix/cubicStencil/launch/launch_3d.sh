#!/bin/bash


EXE_NAME="FDTD3d"
EXE_SUFFIX=".out"
EXE=$EXE_NAME$EXE_SUFFIX

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=../results/$EXE_NAME  #-$TIME_STAMP

SCRIPT="../FDTD3d/${EXE} --timesteps=20 --kernel2 --radius=" # --dimx=216 --dimy=216 --dimz=216

for i in {3..3..1}
do
NAME=$OUTPUT-r$i
sbatch \
    -o $NAME.txt \
    -p skylake \
    --gres=gpu:rtx_2080_ti:1 \
    --exclusive \
    --wrap="${SCRIPT}${i}"
done