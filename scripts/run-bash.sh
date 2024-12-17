#!/bin/bash

# touch ../output/errors/batch-run.out

LOG="../output/errors/batch-run.out"
echo "\n#############################" >> $LOG
date >> $LOG
echo "$@" >> $LOG

ARCHS=("vgg16" "resnet18")
DATASETS=("CIFAR10" "MNIST") # 
# for arch in ${ARCHS[@]}; do for dataset in ${DATASETS[@]}; do
    echo $arch $dataset >> ../output/errors/batch-run.out
    # python prune.py --lr=.001 --arch=$2 --dataset=$3 --gpu=$6 \
    #     --train_epochs=$4 --control_at_iter=-1 --control_at_epoch=2 \
    #     --control_at_layer="2" --experiment_type=$1 --pretrained=False \
    #     --acc_thrd=$4 --imp_total_iter=$5 &

    python prune.py --lr=.001 --arch=$2 --dataset=$3  --gpu=$6\
        --train_epochs=$4 --control_at_iter=1 --control_at_epoch=2 \
        --control_at_layer="2" --experiment_type=$1 --pretrained=False \
        --acc_thrd=$4 --imp_total_iter=$5 &
# done; done;
