#!/bin/bash

diagnosis=$1
mil=$2
pretrain=$3
mode=$4
gpuN=$5

folds=("0" "1" "2" "3" "4")

for i in ${folds[@]}; do

    echo "==================================================="
    echo "fold      : $i/4"
    echo "diagnosis : $diagnosis"
    echo "milType   : $mil"
    echo "pretrain  : $pretrain"
    echo "mode      : $mode"
    echo "==================================================="
    
    python3 ./train.py \
        --mil $mil \
        --mode $mode \
        --pretrain $pretrain\
        --foldN $i \
        --diagnosis $diagnosis \
        --gpuN $gpuN

done
