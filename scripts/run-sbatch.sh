#!/bin/bash

# touch ../output/errors/batch-run.out

LOG="../output/errors/batch-run.out"
echo "\n #############################" >> $LOG
date >> $LOG
echo "$@" >> $LOG

ARCHS=("vgg16" "resnet18")
DATASETS=("CIFAR10" "MNIST") # 
# for arch in ${ARCHS[@]}; do for dataset in ${DATASETS[@]}; do
# echo $arch $dataset >> ../output/errors/batch-run.out
sbatch experiments.slurm --yaml_config=config.ini
# done; done;
