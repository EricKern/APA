#!/bin/bash


EXE_NAME="FDTD3d"
EXE_SUFFIX=""
EXE=$EXE_NAME$EXE_SUFFIX

TIME_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=../results/2-$EXE_NAME-$TIME_STAMP
DIM=220

# compute-sanitizer --destroy-on-device-error kernel ./${EXE} --dimx=$DIM --dimy=$DIM --dimz=$DIM --radius=5 --kernel2
./${EXE} --dimx=$DIM --dimy=$DIM --dimz=$DIM --radius=2 --kernel2