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
import math
import time
from torch.optim.lr_scheduler  import MultiStepLR

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

### An object which handles training, pruning, and testing operations
class Manager(object):
    ### Data loading and makes a pruner instance
    def __init__(self, args, pruner, train_data_loader, val_data_loader, test_data_loader, task_num):
        print("Changes check")
        self.args = args
        self.cuda = args.cuda
        self.task_num = task_num
        self.logsoftmax = nn.LogSoftmax()
        self.criterion = nn.CrossEntropyLoss()

        ### Set up data loader, criterion, and pruner.
        self.train_data_loader = train_data_loader
        self.test_data_loader =  test_data_loader
        self.val_data_loader = self.test_data_loader
        
        self.pruner = pruner
        

    

       
    def eval(self):
        """Performs evaluation."""
        print("Task number in Eval: ", self.task_num, " ", self.pruner.task_num)

        print("Applying dataset mask for current dataset")
        self.pruner.apply_mask()
    
        self.pruner.model.eval()
        error_meter = None

        print('Performing eval...', flush=True)
        
        for batch, label in self.test_data_loader:
            if self.cuda:
                batch = batch.cuda()
                label = label.cuda()
    
            with torch.no_grad():
                output = self.pruner.model(batch)
    
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
                                    
        ### Note: After the first task, batchnorm and bias throughout the network are frozen, this is what train_nobn() refers to
        if self.task_num > 0:
            self.pruner.model.train_nobn()
        else:
            self.pruner.model.train()
        return errors


    ### Train the model for the current task, using all past frozen weights as well
    def train(self, epochs, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_model_acc = best_accuracy
        test_error_history = []

        if self.args.cuda:
            self.pruner.model = self.pruner.model.cuda()

        # Get optimizer with correct params.
        params_to_optimize = self.pruner.model.parameters()
        optimizer = optim.SGD(params_to_optimize, lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=self.args.Milestones, gamma=self.args.Gamma)    

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))
            print("Learning rate:", optimizer.param_groups[0]['lr'])
            if self.task_num > 0:
                self.pruner.model.train_nobn()
                print("No BN in training loop")
            else:
                self.pruner.model.train()
                
            for x, y in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx), disable=True):
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                x = Variable(x)
                y = Variable(y)
    
                # Set grads to 0.
                self.pruner.model.zero_grad()
        
                # Do forward-backward.
                output = self.pruner.model(x)
    
                # print("Forward done")
                self.criterion(output, y).backward()

                # Set frozen param grads to 0.
                self.pruner.make_grads_zero()

                # Update params.
                optimizer.step()

                # Set pruned weights to 0.
                self.pruner.make_pruned_zero()
                            
            scheduler.step()
            
            test_errors = self.eval()
            test_error_history.append(test_errors)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.


            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'test_error_history': test_error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and best_model_acc < test_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_model_acc, test_accuracy))

                best_model_acc = test_accuracy
                self.save_model(savename)

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_model_acc, best_model_acc))
        print('-' * 16)
       


    ### Collect activations, calculate connectivities for filters and layers, then saves them in the checkpoint
    def calc_conns(self, savename = ""):
        """Calculating Connectivities."""
        self.pruner.calc_conns()
        ### Commented out to save storage space, but these will save the connectivity data as their own files
        # np.save(os.path.join(self.args.save_prefix, "conns.npy"), self.pruner.conns)
        # np.save('./conn_aves.npy', self.pruner.conn_aves)
        
        ### Saving model, save_model() will fetch and save the latest connectivity data in self.pruner
        self.save_model(savename)



    ### Call for the pruner to prune the model
    def prune(self, prune_savename=""):
        """Perform pruning."""
        print('Pre-prune eval:')
        self.eval()

        self.pruner.prune()
        self.check(True)

        print('\nPost-prune eval:')
        errors = self.eval()

        accuracy = 100 - errors[0]  # Top-1 accuracy.
        self.save_model(prune_savename)

        # Do final finetuning to improve results on pruned network.
        if self.args.finetune_epochs:
            print('Doing some extra finetuning...')
            self.train(self.args.finetune_epochs, save=True,
                       savename=prune_savename, best_accuracy=0)
        print('-' * 16)
        print('Pruning summary:')
        self.check(True)
        print('-' * 16)
        print("\n\n\n\n\n\n\n")

    ### Just checks how many parameters per layer are now 0 post-pruning
    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.pruner.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))




    ### Saves a checkpoint of the model
    def save_model(self, savename):
        """Saves model to file."""

        # Prepare the ckpt.
        ckpt = {
            'args': self.args,
            'composite_mask': self.pruner.composite_mask,
            'all_task_masks': self.pruner.all_task_masks,
            'conns' : self.pruner.conns,
            'conn_aves' : self.pruner.conn_aves,
            'model': self.pruner.model,
        }

        # Save to file.
        torch.save(ckpt, savename)




    
    ### After each task, this is called to update the task number and have the pruner update necessary information
    def increment_task(self):
            self.task_num += 1
            self.pruner.increment_task()





    
    
        

