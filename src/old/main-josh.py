"""Main entry point for doing all pruning-related stuff. Adapted from https://github.com/arunmallya/packnet/blob/master/src/main.py"""
from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import networks as net
import utils as utils
from prune import SparsePruner
from manager import Manager
import time
from torch.optim.lr_scheduler  import MultiStepLR

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch', choices=['vgg16', 'vgg16bn', 'resnet18', 'densenet121'], help='Architectures')
FLAGS.add_argument('--mode', choices=['t','c','p','e','all'], default='all', help='Run mode: train, calc. conns, prune, finetune, or evaluate')
FLAGS.add_argument('--dataset', type=str, choices=['pmnist', '6splitcifar', '11splitcifar'], default='6splitcifar', help='Name of dataset')
FLAGS.add_argument('--single_task', action='store_true', default=False, help='Run only the current task')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--num_outputs', type=int, default=-1, help='Num outputs for dataset')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--cores', type=int, default=4, help='Number of CPU cores.')
# Other.
# Paths.
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='', help='Location to save model')
# Training options.
FLAGS.add_argument('--train_epochs', type=int, default=2, help='Number of epochs to train for')
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--batch_size', type=int, default=512, help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
FLAGS.add_argument('--Milestones', nargs='+', type=float, default=[30,60,90])
FLAGS.add_argument('--Gamma', type=float, default=0.1)   
# Pruning options.
FLAGS.add_argument('--prune_method', type=str, default='sparse', choices=['sparse'], help='Pruning method to use')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.5, help='% of neurons to prune per layer')
FLAGS.add_argument('--finetune_epochs', type=int, default=2, help='Number of epochs to finetune for after pruning')
FLAGS.add_argument('--freeze_perc', type=float, default=0.0)                   
FLAGS.add_argument('--num_freeze_layers', type=int, default=0)     
FLAGS.add_argument('--freeze_order', choices=['top','bottom', 'random'], default=['top'],help='Order of selection for layer freezing, by connectivity')





###################################################################################################################################################
###
###     Main function
###
###################################################################################################################################################


def main():
    args = FLAGS.parse_args()
    ### Early termination conditions
    if args.prune_perc_per_layer <= 0:
        print("non-positive prune perc",flush = True)
        return    
    torch.cuda.set_device(0)
    
    ### Currently only uses 6splitcifar, which will load cifar10 as task 0, and then 5 splits of cifar100 for tasks 1-5
    if args.dataset == "pmnist":
        taskset = [*range(0,10,1)]
    elif args.dataset == "6splitcifar":
        taskset = [*range(0,6,1)]
    elif args.dataset == "11splitcifar":
        taskset = [*range(0,11,1)]
    else: 
        print("Incorrect dataset name for args.dataset")
        return 0
     
     
    ###################
    ##### Prepare Ckpt
    ###################
    ### The general file structure is that each task has its own subdirectory, before pruning the checkpoint is saved as trained.pt, after pruning it's saved as final.pt
    ###    and final.pt is used when loading the final checkpoint from the previous task, or the initial checkpoint if starting on task 0 (it's saved as task -1 in the directory)
    args.save_prefix = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(args.task_num))
    previoustaskpath = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(args.task_num-1))
    os.makedirs(args.save_prefix, exist_ok = True)
    os.makedirs(previoustaskpath, exist_ok = True)

    trainedpath = os.path.join(args.save_prefix, "trained.pt")
    initpath = os.path.join(previoustaskpath, "final.pt")

    if os.path.isfile(initpath) == False and args.task_num == 0:
        print("initializing model",flush = True)
        utils.init_dump(args.arch, initpath)
    
    if os.path.isfile(os.path.join(previoustaskpath,"final.pt")) == True and (args.mode == "t" or args.mode == "all"):
        ckpt = torch.load(os.path.join(previoustaskpath,"final.pt"))
    elif os.path.isfile(os.path.join("../checkpoints/",str(args.dataset),str(args.prune_perc_per_layer),str(args.run_id),str((len(taskset)-1)), "final.pt")) == True and args.mode == "e":
        ckpt = torch.load(os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str((len(taskset)-1)), "final.pt"))
    elif os.path.isfile(trainedpath) == True and (args.mode == "p" or args.mode == "c"):
        ckpt = torch.load(trainedpath)
    else:
        print("No checkpoint file found")
        return 0
        
    ### The checkpoint contains the model state, composite mask for tracking what weights were frozen for which tasks, all_task_masks for weight sharing, and connectivity information
    model = ckpt['model']
    composite_mask = ckpt['composite_mask']
    all_task_masks = ckpt['all_task_masks']
    conns = ckpt['conns']
    conn_aves = ckpt['conn_aves']

    print("#######################################################################")
    print("Finished Loading Checkpoint")
    print("Composite Mask keys: ", composite_mask.keys())
    print("All task Masks keys: ", all_task_masks.keys())
    print("Num outputs: ", args.num_outputs)
    print("Conns keys: ", conns.keys())
    print("Conn_aves keys: ", conn_aves.keys())
    print("Dataset: " + str(args.dataset))
    print("#######################################################################")

    if args.cuda:
        model = model.cuda()

    ### Initialize a pruner class object which handles connectivity and pruning functions
    pruner = SparsePruner(args, model, all_task_masks, composite_mask, conns, conn_aves)

    ### Looping over remaining tasks
    for task in taskset[args.task_num:]:
        args.save_prefix = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(task))
        os.makedirs(args.save_prefix, exist_ok = True)

        trainedpath = os.path.join(args.save_prefix, "trained.pt")
        finetunedpath = os.path.join(args.save_prefix, "final.pt")

        
        # Create the manager object.
        train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="train")
        test_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="test")
        val_data_loader = test_data_loader

        ### This is for producing and setting the classifier layer for a given task's # classes
        pruner.model.add_dataset(str(task), args.num_outputs)
        pruner.model.set_dataset(str(task))
                
        pruner.testloader = test_data_loader
        manager = Manager(args, pruner, train_data_loader, val_data_loader, test_data_loader, task)
        print("Manager created")


        ### train for new task
        if  args.mode == "t" or args.mode == "all":
            print("Training", flush = True)
            manager.train(args.train_epochs, save=True, savename=trainedpath)
        
        ### calculate connectivity scores
        if  args.mode == "c" or args.mode == "all":
            print("Calculate Connectivities")
            manager.calc_conns(savename=trainedpath) 
        
        ### Prune unecessary weights or nodes
        if  args.mode == "p" or args.mode == "all":
            print("Pruning", flush = True)
            manager.prune(prune_savename=finetunedpath)
            
        ### Evaluate performance on task
        if  args.mode == "e":
            print("Evaluating Accuracy")
            manager.eval()

        ### When the task ends, if there are more tasks to run then we update as needed and proceed to the next task
        if args.single_task == False and args.mode == "all":
            manager.increment_task()
            manager.save_model(savename=finetunedpath)
        elif args.mode == "p":
            manager.increment_task()
            manager.save_model(savename=finetunedpath)
            return 0
        else: 
            return 0

    

    
    
        

    
if __name__ == '__main__':
    main()
