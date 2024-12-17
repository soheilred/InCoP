"""
Handles all the pruning and connectivity. Pruning steps are adapted from: https://github.com/arunmallya/packnet/blob/master/src/prune.py
Connectivity steps and implementation of connectivity into the pruning steps are part of our contribution
"""
from __future__ import print_function

import collections
import time
import copy
import torch
import random
import multiprocessing
import torch.nn as nn
import numpy             as np

# Custom imports
from josh_utils                   import activations, corr


class SparsePruner(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the pruner to explicitly show which arguments are used by it
    def __init__(self, args, model, all_task_masks, composite_mask,
                        train_loader, test_loader, conns, conn_aves):
        self.args = args
        self.model = model 
        self.prune_perc = args.prune_perc_per_layer
        self.freeze_perc = args.freeze_perc
        self.num_freeze_layers = args.num_freeze_layers
        self.freeze_order = args.freeze_order
        self.train_bias = args.train_biases
        self.train_bn = args.train_bn 
        
        self.trainloader = train_loader 
        self.testloader = test_loader 
        
        self.conns = conns
        self.conn_aves = conn_aves
        
        self.composite_mask = composite_mask
        self.all_task_masks = all_task_masks 


        self.task_num = self.args.task_num
        print("current index is: " + str(self.task_num))
    
    
    
     
    """
    ###########################################################################################
    #####
    #####  Connectivity Functions
    #####
    #####  Use: Gets the connectivity between each pair of convolutional or linear layers. 
    #####       The primary original code for our published connectivity-based freezing method
    #####
    ###########################################################################################
    """

      
    def calc_conns(self):
        self.task_conns = {}
        self.task_conn_aves = {}        
        
        
        ### Record the indices of all adjacent pairs of layers in the shared network
        ### This numbering method reflects the "layer index" in Figs. 2-4 of the accompanying paper
        parents = []
        children = []
        i = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear):
                if (i == 0):
                    parents.append(module_idx)
                    i += 1
                else:
                    parents.append(module_idx)
                    children.append(module_idx)

        for key_id in range(0,len(children)): 
            self.task_conn_aves[parents[key_id]], self.task_conns[parents[key_id]] = self.calc_conn([parents[key_id]], [children[key_id]], key_id)
            
        self.conn_aves[self.args.task_num] = self.task_conn_aves
        self.conns[self.args.task_num] = self.task_conns


   
    
    def calc_conn(self, parent_key, children_key, key_id):
        self.model.eval()
    
        # Obtain Activations
        print("----------------------------------")
        print("Collecting activations from layers")
    
        p1_op = {}
        p1_lab = {}
        c1_op = {}
        c1_lab = {}    

        unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
        act         = {}
        lab         = {}
    

        ### Get activations and labels from the function in utils prior to calculating connectivities
        for item_key in unique_keys:
            act[item_key], lab[item_key] = activations(self.testloader, self.model, self.args.cuda, item_key)

        for item_idx in range(len(parent_key)):
            p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
            p1_lab[str(item_idx)] = copy.deepcopy(lab[parent_key[item_idx]]) 
            c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])
            c1_lab[str(item_idx)] = copy.deepcopy(lab[children_key[item_idx]])
        
        del act, lab

       
        print("----------------------------------")
        print("Begin Execution of conn estimation")
    
        parent_aves = []
        p1_op = np.asarray(list(p1_op.values())[0])
        p1_lab = np.asarray(list(p1_lab.values())[0])
        c1_op = np.asarray(list(c1_op.values())[0])
        c1_lab = np.asarray(list(c1_lab.values())[0])
    
        task = self.args.task_num


        for c in list(np.unique(np.asarray(p1_lab))):
            p1_op[p1_lab == c] -= np.mean(p1_op[p1_lab == c])
            p1_op[p1_lab == c] /= np.std(p1_op[p1_lab == c])

            c1_op[c1_lab == c] -= np.mean(c1_op[c1_lab == c])
            c1_op[c1_lab == c] /= np.std(c1_op[c1_lab == c])

        """
        Code for averaging conns by parent prior by layer
        """
        
        # parent_class_aves = []

        parents_by_class = []
        # parents_aves = []
        # conn_aves = []
        # parents = []
        
        # for c in list(np.unique(np.asarray(p1_lab))):
        #     p1_class = p1_op[np.where(p1_lab == c)]
        #     c1_class = c1_op[np.where(c1_lab == c)]

        #     pool = multiprocessing.Pool(self.args.cores)

        #     ### Parents is a 2D list of all of the connectivities of parents and children for a single class
        #     parents = pool.starmap(corr, [(p1_class[:,p], c1_class[:,:]) for p in list(range(len(p1_op[0])))], chunksize = 8)

        #     pool.close()
        #     pool.join()

        #     ### This is a growing list of each p-c connectivity for all activations of a given class
        #     ###     The dimensions are (class, parent, child)
        #     parents_by_class.append(parents)

        # ### This is the final array of appended class sets of parent-child connectivities
        # ### Shape should be 10x64x64 for layer 1 in cifar10
        parents_by_class = np.asarray(parents_by_class)
        
        ### Averages all classes, since all class priors are the same for cifar10 and 100
        conn_aves = np.mean(parents_by_class[:], axis=0)
        
        ### Then average over the parents and children to get the layer-layer connectivity
        layer_ave = np.mean(conn_aves[:])

        return layer_ave, conn_aves



    
    
    
    
