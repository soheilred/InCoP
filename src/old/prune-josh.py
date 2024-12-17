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
# from utils import activations, corr
from utils import activations


class SparsePruner(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the pruner to explicitly show which arguments are used by it
    def __init__(self, args, model, all_task_masks, composite_mask, conns, conn_aves):
        self.args = args
        self.model = model 
        self.prune_perc = args.prune_perc_per_layer
        self.freeze_perc = args.freeze_perc
        self.num_freeze_layers = args.num_freeze_layers
        self.freeze_order = args.freeze_order
        self.task_num = args.task_num
        print("current index is: " + str(self.task_num))
        
        self.testloader = None 
        
        self.conns = conns
        self.conn_aves = conn_aves
        
        ### The composite mask stores the task number for which every weight was frozen, or if they are unfrozen the number is the current task
        self.composite_mask = composite_mask
        ### All_task_masks is a dictionary of binary masks, 1 per task, indicating which weights to include when evaluating that task
        self.all_task_masks = all_task_masks 
        

    
     
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
        
        #!# Probably just calculate the activations here before the looping
        
        
        #!# This can still be the same, splitting by parents and children, it just needs to index the acts appropriately rather than generate them each time
        ### Record the indices of all adjacent pairs of layers in the shared network
        ### This numbering method reflects the "layer index" in Figs. 2-4 of the accompanying paper
        parents = []
        children = []
        i = 0
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if (i == 0):
                    parents.append(module_idx)
                    i += 1
                else:
                    parents.append(module_idx)
                    children.append(module_idx)
        children.append(-1)
        for key_id in range(0,len(parents)): 
            self.task_conn_aves[parents[key_id]], self.task_conns[parents[key_id]] = self.calc_conn([parents[key_id]], [children[key_id]], key_id)
            
        self.conn_aves[self.task_num] = self.task_conn_aves
        self.conns[self.task_num] = self.task_conns


   ### This was following an implementation by Madan which allowed for parallelization, where we calculate connectivity for one pair of layers at a time
    def calc_conn(self, parent_key, children_key, key_id):
        self.model.eval()
    
        # Obtain Activations
        print("----------------------------------")
        print("Collecting activations from layers")
    
        p1_op = {}
        c1_op = {}
        p1_lab = {}
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
    
        task = self.task_num

        ### Normalizing the activation values per our proof in the previous work
        for label in list(np.unique(np.asarray(p1_lab))):
            # print("Parent mean and stdev: ", np.mean(p1_op[p1_lab == label]), " ", np.std(p1_op[p1_lab == label]))

            p1_op[p1_lab == label] -= np.mean(p1_op[p1_lab == label])
            p1_op[p1_lab == label] /= np.std(p1_op[p1_lab == label])



            c1_op[c1_lab == label] -= np.mean(c1_op[c1_lab == label])
            c1_op[c1_lab == label] /= np.std(c1_op[c1_lab == label])

        """
        Code for averaging conns by parent prior by layer
        """
        
        parent_class_aves = []
        parents_by_class = []
        parents_aves = []
        conn_aves = []
        parents = []
        
        print("p1_op shape: ", p1_op.shape)

        ### Calculating connectivity for each class individually and then averaging over all classes, as per proofs
        for c in list(np.unique(np.asarray(p1_lab))):
            p1_class = p1_op[np.where(p1_lab == c)]
            c1_class = c1_op[np.where(c1_lab == c)]

            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            coefs = np.corrcoef(p1_class, c1_class, rowvar=False)
            
            parents = []
            for i in range(0, len(p1_class[0])):
                parents.append(coefs[i, len(p1_class[0]):])
            parents = np.abs(np.asarray(parents))

            ### This is a growing list of each p-c connectivity for all activations of a given class
            ###     The dimensions are (class, parent, child)
            parents_by_class.append(parents)
        
        ### Averages all classes, since all class priors are the same for cifar10 and 100
        conn_aves = np.mean(np.asarray(parents_by_class), axis=0)
        
        ### Then average over the parents and children to get the layer-layer connectivity
        layer_ave = np.mean(conn_aves)

        return layer_ave, conn_aves



    
    
    
    
    
    """
    ##########################################################################################################################################
    Pruning Functions
    ##########################################################################################################################################
    """
    ### Goes through and calls prune_mask for each layer and stores the results
    ### Then applies the masks to the weights
    def prune(self):
        print('Pruning for dataset idx: %d' % (self.task_num))
        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
    
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### Get the pruned mask for the current layer
                mask = self.pruning_mask(module.weight.data, self.composite_mask[module_idx], module_idx)
                self.composite_mask[module_idx] = mask.cuda()

                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.composite_mask[module_idx].gt(self.task_num)] = 0.0
                self.all_task_masks[self.task_num][0][module_idx][mask.gt(self.task_num)] = 0

                
        print("\nFOR TASK %d:", self.task_num)
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_frozen = weight[self.composite_mask[layer_idx].eq(self.task_num)].numel()
                print('Layer #%d: Frozen %d/%d (%.2f%%)' %
                      (layer_idx, num_frozen, num_params, 100 * num_frozen / num_params))                

      
 
 
 
 
    def pruning_mask(self, weights, composite_mask_in, layer_idx):
        """
            Ranks prunable filters by magnitude. Sets all below kth to 0.
            Returns pruned mask.
        """

        composite_mask = composite_mask_in.clone().detach().cuda()

        filter_weights = weights
        filter_composite_mask = composite_mask.eq(self.task_num)
        tensor = weights[filter_composite_mask]

        abs_tensor = tensor.abs()


        """
            Code for increasing the freezing percent of a given layer based on connectivity
        """

        prune_rank = round(self.prune_perc * abs_tensor.numel())
        connlist = self.conn_aves[self.task_num]


        max_n_layers_indices = np.argsort(list(connlist.values()))
        max_n_keys = np.asarray(list(connlist.keys()))[list(max_n_layers_indices)]
        random_idxs = np.copy(max_n_keys)
        np.random.shuffle(random_idxs)
        
        # ### Apply freezing if the index is selected based on connectivity, otherwise prune at the baseline rate.
        # if self.freeze_order == "top" and (layer_idx in max_n_keys[-self.num_freeze_layers:]):
        #     prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        # elif self.freeze_order == "bottom" and (layer_idx in max_n_keys[:self.num_freeze_layers]):
        #     prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        # elif self.freeze_order == "random" and (layer_idx in random_idxs[:self.num_freeze_layers]):
        #     prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        
        prune_value = abs_tensor.view(-1).cpu().kthvalue(prune_rank)[0]

        remove_mask = torch.zeros(weights.shape)

        remove_mask[filter_weights.abs().le(prune_value)]=1
        remove_mask[composite_mask.ne(self.task_num)]=0

        composite_mask[remove_mask.eq(1)] = self.task_num + 1
        mask = composite_mask
        
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.gt(self.task_num).sum(), tensor.numel(),
              100 * mask.gt(self.task_num).sum() / tensor.numel(), weights.numel()))

        return mask
        
        
        
        
    
        
        
    
    """
    ##########################################################################################################################################
    Update Functions
    ##########################################################################################################################################
    """

    ### During training this is called to avoid storing gradients for the frozen weights, to prevent updating
    ### This is unaffected in the shared masks since shared weights always have the current index unless frozen
    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        # assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.task_num)] = 0
                if self.task_num>0 and module.bias is not None:
                    module.bias.grad.data.fill_(0)
                    
            elif 'BatchNorm' in str(type(module)) and self.task_num>0:
                    # Set grads of batchnorm params to 0.
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    ### Set all pruned weights to 0
    ### This is just a prune() but with pre-calculated masks
    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        # assert self.current_masks
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                module.weight.data[layer_mask.gt(self.task_num)] = 0.0


    ### Applies appropriate mask to recreate task model for inference
    def apply_mask(self):
        """To be done to retrieve weights just for a particular dataset"""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = -100

                ### Any weights which weren't frozen in one of the tasks before or including task # dataset_idx are set to 0
                for i in range(0, self.task_num+1):
                    if i == 0:
                        mask = self.all_task_masks[i][0][module_idx].cuda()
                    else:
                        mask = mask.logical_or(self.all_task_masks[i][0][module_idx].cuda())
                weight[mask.eq(0)] = 0.0
   


    """
        Turns previously pruned weights into trainable weights for
        current dataset.
        Also updates task number and prepares new task mask
    """
    def increment_task(self):
        self.task_num += 1
                
        ### Creates the task-specific mask during the initial weight allocation
        task_mask = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                task = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    task = task.cuda()
                task_mask[module_idx] = task

        ### Initialize the new tasks' inclusion map with all 1's
        self.all_task_masks[self.task_num] = [task_mask]

        print("Exiting finetuning mask")
