#!/bin/bash

# touch ../output/errors/batch-run.out

PID=$(echo $$)
echo "main process: $PID"

CNTLOG="../output/errors/cnt-${PID}.out"
NOCNTLOG="../output/errors/nocnt-${PID}.out"
echo "#############################" >> $CNTLOG
echo "#############################" >> $NOCNTLOG
date >> $CNTLOG
date >> $NOCNTLOG
echo "$@" >> $CNTLOG
echo "$@" >> $NOCNTLOG

# ARCHS=("vgg16" "resnet18")
# DATASETS=("CIFAR10" "MNIST") # 
# for arch in ${ARCHS[@]}; do for dataset in ${DATASETS[@]}; do
    # echo $arch $dataset >> ../output/errors/batch-run.out
    # python prune.py --lr=.001 --arch=$2 --dataset=$3 --gpu=$6 \
    #     --train_epochs=$4 --control_at_iter=-1 --control_at_epoch=2 \
    #     --control_at_layer="2" --experiment_type=$1 --pretrained=False \
    #     --acc_thrd=$4 --imp_total_iter=$5 &

    # python prune.py --lr=.001 --arch=$2 --dataset=$3  --gpu=$6\
    #     --train_epochs=$4 --control_at_iter=1 --control_at_epoch=2 \
    #     --control_at_layer="2" --experiment_type=$1 --pretrained=False \
    #     --acc_thrd=$4 --imp_total_iter=$5 &

cd ../src/
# conda activate ib
source /home/soheil/anaconda3/bin/activate tf
python prune.py $@ --exper_gpu_id=0 --control_on=1 &>> $CNTLOG &
CNTPID=$(echo $!)
python prune.py $@ --exper_gpu_id=1 --control_on=0 &>> $NOCNTLOG &
NOCNTPID=$(echo $!)
echo $CNTPID >> $CNTLOG
echo $NOCNTPID >> $NOCNTLOG
echo $CNTPID
echo $NOCNTPID
# done; done;
