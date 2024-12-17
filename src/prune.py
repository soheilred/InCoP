import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
# import torchvision.datasets as datasets
import torch.nn.init as init
import sys
from collections import OrderedDict

import utils
from data_loader import Data
from similarity import Similarity
import plot_tool as ptool
from network import Network, train, test, warm_up
from activation import Activations
import constants as C
import logging
import logging.config

log = logging.getLogger("sampleLogger")


class Controller:
    def __init__(self, args):
        """Control the IMP's connectivity.
        """
        self.c_type = args.control_type
        self.c_layers = args.control_layer
        self.c_iter = args.control_iteration
        self.c_epoch = args.control_epoch


class Pruner:
    def __init__(self, args, model, act, controller=None):
        """Prune the network.
        Parameters
        ----------
        model: Network
            The network to be pruned
        prune_percent: int
            The percent to which each layer of the network is pruned
        train_dataloader: dataloader
        test_dataloader: dataloader
        composite_mask:
            The composite mask stores the task number for which every weight was
            frozen, or if they are unfrozen the number is the current task 
        all_task_masks: dict
            A dictionary of binary masks, 1 per task, indicating
            which weights to include when evaluating that task 
        """
        self.model = model
        self.mask = {}
        self.args = args
        self.prune_perc = args.prune_ratio
        self.corrs = []
        self.act = act
        self.controller = controller
        # self.correlation = correlation
        self.num_layers = 0
        self.comp_level = np.zeros(args.exper_imp_total_iter, float)
        self.all_acc = np.zeros([args.exper_imp_total_iter, args.net_train_epochs], float)
        self.init_state_dict = None
        # self.init_dump()

    def weight_init(self, m):
        '''Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    def init_lth(self):
        """Prepare the lth object by:
        1. initializing the network's weights
        2. saving the initial state of the network into the object
        3. saving the initial state model on the disk
        4. initializing the masks according to the layers size
        """

        # Weight Initialization
        if self.args.net_pretrained == "False":
            self.model.apply(self.weight_init)

        # Copying and Saving Initial State
        self.init_state_dict = copy.deepcopy(self.model.state_dict())
        run_dir = utils.get_run_dir(self.args)
        utils.save_model(self.model, run_dir, "initial_model.pth.tar")

        # Making Initial Mask
        self.init_mask()
        return self.init_state_dict

    def count_layers(self):
        count = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                count = count + 1
        return count

    def init_mask(self):
        """Make an empty mask of the same size as the model."""
        # self.num_layers = self.count_layers()
        # self.mask = [None] * self.num_layers
        # layer_id = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # tensor = param.data.cpu().numpy()
                # self.mask[layer_id] = np.ones_like(tensor)
                # self.mask[layer_id] = param.new_ones(param.size(),
                #                                      dtype=torch.bool)
                self.mask[name[:-7]] = torch.ones_like(param.data,
                                                       dtype=torch.bool)
                # layer_id += 1

    def reset_weights_to_init(self, initial_state_dict):
        """Reset the remaining weights in the network to the initial values.
        """
        # step = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() > 1:
                weight_dev = param.device
                param.data = (self.mask[name[:-7]] * initial_state_dict[name]).to(weight_dev)
                # step += 1

            if "bias" in name:
                param.data = initial_state_dict[name]

    def prune_by_grads(self):
        # grads = self.act.get_gradient()[0]

        # for module_idx, module in enumerate(self.model.named_modules()):
        #     if isinstance(module[1], nn.Conv2d) or \
        #                  isinstance(module[1], nn.Linear):
        #         if "downsample" in module[0]:
        #             continue

        #         if layer_id > 0:
        #             layer_id += 1
        #             continue
        #        weight = module[1].weight.data

        for name, param in self.model.named_parameters():

            # We do not prune bias term
            if 'weight' in name and param.dim() > 1:
                weight_dev = param.device
                # grad = grads[name].to(weight_dev)
                grad = param.grad
                weight = param.data
                # tensor = grad * weight
                tensor = grad
                alive = tensor[tensor.nonzero(as_tuple=True)]  # flattened array of nonzero values
                # percentile_val = torch.quantile(alive.abs(),
                #                                   self.prune_perc).item()
                percentile_val = np.quantile(alive.abs().detach().cpu().numpy(),
                                             self.prune_perc)
                weight_dev = param.device
                new_mask = torch.where(tensor.abs() < percentile_val, 0,
                                       self.mask[name[:-7]])
                new_mask = new_mask.type(torch.bool).to(weight_dev)

                # Apply new weight and mask
                weight = (weight * new_mask).to(weight_dev)
                self.mask[name[:-7]] = new_mask

    def prune_by_correlation(self):
        corrs = self.act.get_correlations()[-1]
        layers_idx = [elem[1] for elem in self.act.get_layers_idx()]

        # for module_idx, module in enumerate(self.model.named_modules()):
        #     if isinstance(module[1], nn.Conv2d) or \
        #                  isinstance(module[1], nn.Linear):
        #         if "downsample" in module[0]:
        #             continue

        #         if layer_id > 0:
        #             layer_id += 1
        #             continue

        for name, param in self.model.named_parameters():

            # We do not prune bias term
            if 'weight' in name and param.dim() > 1:
                weight_dev = param.device
                weight = param.data
                # if corr existst for the layer
                if ((name[:-7] in layers_idx) and
                        (layers_idx.index(name[:-7]) > 0) and
                        (layers_idx.index(name[:-7]) < len(layers_idx) - 1)):
                    # -1 because the first correlation is between 1 and 2!
                    idx = layers_idx.index(name[:-7]) - 1
                    # fixing the dimension of correlation for multiplication
                    kernel_size = weight.shape[-1]
                    corr = corrs[idx]
                    # Normalize the correlations
                    corr -= corr.min(1, keepdim=True)[0]
                    corr -= corr.max(1, keepdim=True)[0]
                    corr = corr.repeat([kernel_size, kernel_size, 1, 1])
                    corr = corr.permute(3, 2, 1, 0).to(weight_dev)
                    tensor = corr * weight

                # else, prune based on weights
                else:
                    tensor = weight
                # weight = module[1].weight.data
                alive = tensor[tensor.nonzero(as_tuple=True)]  # flattened array of nonzero values
                # percentile_value = torch.quantile(alive.abs(),
                #                                   self.prune_perc).item()
                percentile_val = np.quantile(alive.abs().detach().cpu().numpy(),
                                             self.prune_perc)
                new_mask = torch.where(tensor.abs() < percentile_val, 0,
                                       self.mask[name[:-7]])
                new_mask = new_mask.type(torch.bool).to(weight_dev)

                # Apply new weight and mask
                weight = (weight * new_mask).to(weight_dev)
                self.mask[name[:-7]] = new_mask

    def prune_by_percentile(self):
        # Calculate percentile value
        # layer_id = 0
        for name, param in self.model.named_parameters():

            if 'weight' in name and param.dim() > 1:
                tensor = param.data
                alive = tensor[tensor.nonzero(as_tuple=True)]  # flattened array of nonzero values
                # percentile_value = torch.percentile(alive, self.prune_perc)
                # percentile_value = torch.quantile(alive.abs(),
                #                                   self.prune_perc).item()
                # Convert Tensors to numpy and calculate to avoid runtime error
                percentile_val = np.quantile(alive.abs().detach().cpu().numpy(),
                                             self.prune_perc)
                weight_dev = param.device
                new_mask = torch.where(tensor.abs() < percentile_val, 0,
                                       self.mask[name[:-7]])
                new_mask = new_mask.type(torch.bool).to(weight_dev)

                # tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                # percentile_value = np.percentile(abs(alive), self.prune_perc)

                # # Convert Tensors to numpy and calculate
                # weight_dev = param.device
                # new_mask = np.where(abs(tensor) < percentile_value, 0,
                #                     self.mask[layer_id])

                # Apply new weight and mask
                param.data = (tensor * new_mask)
                # param.grad *= new_mask
                self.mask[name[:-7]] = new_mask
                # layer_id += 1

    def prune_by_corr_sap(self):
        layer_id = 0
        pivot_param = []
        pivot_mask = []

        for name, param in self.model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                # mask_i = mask.state_dict()[name]
                pivot_param_i = param[self.mask[name[:-7]]].abs()
                pivot_param.append(pivot_param_i.view(-1))
                pivot_mask.append(self.mask[name[:-7]].view(-1))
                layer_id += 1

        pivot_param = torch.cat(pivot_param, dim=0).data.abs()
        pivot_mask = torch.cat(pivot_mask, dim=0)
        # p, q, eta_m, gamma = self.prune_mode[1:] # TODO
        p, q, eta_m, gamma = float(1.0), float(2.0), float(0.), float(1.)
        beta = 0.9
        sparsity_index = {"p": torch.arange(0.1, 1.1, 0.1),
                          "q": torch.arange(1.0, 2.1, 0.1)}
        p_idx = (sparsity_index["p"] == p).nonzero().item()
        q_idx = (sparsity_index["q"] == q).nonzero().item()
        mask_i = pivot_mask
        si = self.make_si_corr(self.model, self.mask, sparsity_index)
        si_i = si[p_idx, q_idx]
        d = mask_i.float().sum().to('cpu')
        m = d * (1 + eta_m) ** (q / (p - q)) * (1 - si_i) ** ((q * p) / (q - p))
        m = torch.ceil(m).long()
        retain_ratio = m / d
        prune_ratio = torch.clamp(gamma * (1 - retain_ratio), 0, beta)
        num_prune = torch.floor(d * prune_ratio).long()
        pivot_value = torch.sort(pivot_param.view(-1))[0][num_prune]

        # new_mask = [None] * self.num_layers
        # layer_id = 0
        # new_mask = OrderedDict()
        for name, param in self.model.named_parameters():

            if 'weight' in name and param.dim() > 1:
                # tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)]
                # percentile_value = np.percentile(abs(alive), self.prune_perc)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                # new_mask = np.where(abs(tensor) < percentile_value, 0,
                #                     self.mask[layer_id])

                pivot_mask = (param.data.abs() < pivot_value).to(weight_dev)
                new_mask = torch.where(pivot_mask, False, self.mask[name[:-7]])

                # Apply new weight and mask
                # param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                param.data = torch.where(new_mask.to(param.device), param.data,
                                         torch.tensor(0, dtype=torch.float,
                                                      device=param.device))

                param.grad.mul_(new_mask)
                self.mask[name[:-7]] = new_mask
                # layer_id += 1

    def make_si_corr(self, model, mask, si_dict):
        corrs = self.act.get_correlations()[-1]
        layers_idx = [elem[1] for elem in self.act.get_layers_idx()]
        sparsity_index = OrderedDict()
        param_all = []
        mask_all = []
        # layer_id = 0
        for name, param in model.state_dict().items():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                weight_dev = param.device
                weight = param.data
                if ((name[:-7] in layers_idx) and
                        (layers_idx.index(name[:-7]) > 0) and
                        (layers_idx.index(name[:-7]) < len(layers_idx) - 1)):
                    idx = layers_idx.index(name[:-7]) - 1
                    # tensor = corr.repeat(fix-dim) * weight
                    kernel_size = weight.shape[-1]
                    # correlation = correlations[layer_id - 1]
                    corr = corrs[idx].repeat([kernel_size, kernel_size, 1, 1])
                    corr = corr.permute(3, 2, 1, 0).to(weight_dev)
                    tensor = corr * weight

                # else, prune based on weights
                else:
                    tensor = weight
                # weight = module[1].weight.data
 
                # param_all.append(tensor.view(-1))
                param_all.append(tensor.flatten())
                mask_all.append(self.mask[name[:-7]].view(-1))
                # layer_id += 1
        param_all = torch.cat(param_all, dim=0)
        mask_all = torch.cat(mask_all, dim=0)
        sparsity_index_i = []
        for i in range(len(si_dict["p"])):
            for j in range(len(si_dict["q"])):
                sparsity_index_i.append(self.make_si(param_all, mask_all, -1,
                                                     si_dict["p"][i],
                                                     si_dict["q"][j]))
        sparsity_index_i = torch.tensor(sparsity_index_i)
        sparsity_index = sparsity_index_i.reshape(
            (len(si_dict["p"]), len(si_dict["q"]), -1))

        return sparsity_index

    def prune_by_sap(self):
        # layer_id = 0
        pivot_param = []
        pivot_mask = []

        for name, param in self.model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                # mask_i = mask.state_dict()[name]
                pivot_param_i = param[self.mask[name[:-7]]].abs()
                pivot_param.append(pivot_param_i.view(-1))
                pivot_mask.append(self.mask[name[:-7]].view(-1))
                # layer_id += 1

        pivot_param = torch.cat(pivot_param, dim=0).data.abs()
        pivot_mask = torch.cat(pivot_mask, dim=0)
        # p, q, eta_m, gamma = self.prune_mode[1:] # TODO
        p, q, eta_m, gamma = self.args.prune_p, self.args.prune_q, float(0.), float(1.)
        beta = 0.9
        sparsity_index = {"p": torch.arange(0.1, 1.1, 0.1),
                          "q": torch.arange(1.0, 2.1, 0.1)}
        p_idx = (sparsity_index["p"] == p).nonzero().item()
        q_idx = (sparsity_index["q"] == q).nonzero().item()
        mask_i = pivot_mask
        si = self.make_si_(self.model, self.mask, sparsity_index)
        # I(w_t)
        si_i = si[p_idx, q_idx]
        d = mask_i.float().sum().to('cpu')
        m = d * (1 + eta_m) ** (q / (p - q)) * (1 - si_i) ** ((q * p) / (q - p))
        m = torch.ceil(m).long()
        retain_ratio = m / d
        prune_ratio = torch.clamp(gamma * (1 - retain_ratio), 0, beta)
        num_prune = torch.floor(d * prune_ratio).long()
        pivot_value = torch.sort(pivot_param.view(-1))[0][num_prune]

        # new_mask = [None] * self.num_layers
        # layer_id = 0
        # new_mask = OrderedDict()
        for name, param in self.model.named_parameters():

            if 'weight' in name and param.dim() > 1:
                # tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)]
                # percentile_value = np.percentile(abs(alive), self.prune_perc)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                # new_mask = np.where(abs(tensor) < percentile_value, 0,
                #                     self.mask[layer_id])

                pivot_mask = (param.data.abs() < pivot_value).to(weight_dev)
                new_mask = torch.where(pivot_mask, False, self.mask[name[:-7]])

                # Apply new weight and mask
                # param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                param.data = torch.where(new_mask.to(param.device), param.data,
                                         torch.tensor(0, dtype=torch.float,
                                                      device=param.device))

                param.grad.mul_(new_mask)
                self.mask[name[:-7]] = new_mask
                # layer_id += 1

    def make_si_(self, model, mask, si_dict):
        sparsity_index = OrderedDict()
        param_all = []
        mask_all = []
        # layer_id = 0
        for name, param in model.state_dict().items():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                param_all.append(param.view(-1))
                mask_all.append(self.mask[name[:-7]].view(-1))
                # layer_id += 1
        param_all = torch.cat(param_all, dim=0)
        mask_all = torch.cat(mask_all, dim=0)
        sparsity_index_i = []
        for i in range(len(si_dict["p"])):
            for j in range(len(si_dict["q"])):
                sparsity_index_i.append(self.make_si(param_all, mask_all, -1,
                                                     si_dict["p"][i],
                                                     si_dict["q"][j]))
        sparsity_index_i = torch.tensor(sparsity_index_i)
        sparsity_index = sparsity_index_i.reshape(
            (len(si_dict["p"]), len(si_dict["q"]), -1))

        return sparsity_index

    def make_si(self, x, mask, dim, p, q):
        d = mask.to(x.device).float().sum(dim=dim)
        x = x * mask.to(x.device).float()
        si = 1 - (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p) / \
            (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
        si[d == 0] = 0
        si[si == -float('inf')] = 0
        si[torch.logical_and(si > - 1e-5, si < 0)] = 0
        return si

    def prune_once(self, initial_state_dict):

        log.debug(f"Prunning using {self.args.prune_method}")
        if self.args.prune_method == "lth":
            self.prune_by_percentile()

        elif self.args.prune_method == "sap":
            self.prune_by_sap()

        elif self.args.prune_method == "ciap":
            self.prune_by_sap()

        elif self.args.prune_method == "giap":
            self.prune_by_sap()

        else:
            sys.exit("Wrong pruning method!")

        self.reset_weights_to_init(initial_state_dict)

    def make_pruned_zero(self):
        """Set all pruned weights to 0.
            This is just a prune() but with pre-calculated masks
        """
        # assert self.current_masks
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                module.weight.data[layer_mask.gt(self.task_num)] = 0.0

    def apply_mask(self):
        """Applies appropriate mask to recreate task model for inference.
            To be done to retrieve weights just for a particular dataset
        """
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = -100

                # Any weights which weren't frozen in one of the tasks before
                # or including task # dataset_idx are set to 0 
                for i in range(0, self.task_num + 1):
                    if i == 0:
                        mask = self.all_task_masks[i][0][module_idx].cuda()
                    else:
                        mask = mask.logical_or(self.all_task_masks[i][0][module_idx].cuda())
                weight[mask.eq(0)] = 0.0

        self.model.eval()

    def control(self, layers_dim, imp_iter):
        control_corrs = self.corrs
        log.debug(f"apply controller at layer {self.controller.c_layers}")

        # print([corr.shape for corr in control_corrs[-1]])

        # get the weights from previous iteration
        prev_iter_weights = self.get_prev_iter_weights(imp_iter)

        # get connectivity
        connectivity = [(torch.mean(control_corrs[imp_iter - 1][i]) /
                        (layers_dim[i][0] * layers_dim[i + 1][0]))
                        for i in range(len(layers_dim) - 1)]

        # get the coefficient based on connectivity
        for ind in self.controller.c_layers:
            prev_corr = self.correlation_to_weights(control_corrs, layers_dim,
                                                    imp_iter, ind)
            prev_weight = prev_iter_weights[ind]

            # type 1
            if (self.controller.c_type == 1):
                control_weights = abs(prev_corr) # / max(connectivity)

            # type 2
            elif (self.controller.c_type == 2):
                control_weights = torch.mul(prev_corr, prev_weight)

            # type 3
            elif (self.controller.c_type == 3):
                control_weights = abs(connectivity[ind]) / max(connectivity) # * prev_weight

            # type 4
            elif (self.controller.c_type == 4):
                control_weights = abs(prev_corr)
                control_weights = torch.exp(control_weights) /\
                    np.exp(control_weights).sum()

            elif (self.controller.c_type == 5):
                control_weights = torch.exp(abs(prev_corr))

            self.apply_controller(control_weights, ind)

    def apply_controller(self, control_weights, layer_idx):
        idx = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if "downsample" in module[0]:
                    continue

                if (idx == layer_idx):
                    # weight = module[1].weight.detach().cpu().numpy()
                    weight = module[1].weight.data
                    print("network's weight shape", module[0], weight.shape)
                    # mod_weight = weight.cpu().numpy()
                    weight_dev = module[1].weight.device
                    control_weights = control_weights.to(weight_dev)
                    # control_weights = torch.from_numpy(control_weights.astype("float32")).to(weight_dev)
                    new_weight = (weight * control_weights).type(torch.cuda.FloatTensor)
                    # module[1].weight = torch.nn.Parameter(new_weight,
                    #                                        dtype=torch.float,
                    #                                        device=weight_dev)
                    print("control weight", torch.linalg.norm(control_weights))
                    print("old weight", torch.linalg.norm(weight))
                    print("new weight", torch.linalg.norm(new_weight))
                    weight = new_weight
                    break
                idx += 1

    def correlation_to_weights(self, control_corrs, layers_dim, imp_iter, layer_ind):
        # the + 1 is for matching to the connectivity's dimension
        weights = control_corrs[imp_iter - 1][layer_ind - 1]
        # weights = control_corrs[0][layer_ind - 1]
        kernel_size = layers_dim[layer_ind][-1]
        weights = weights.repeat([kernel_size, kernel_size, 1, 1]).\
            permute(3, 2, 1, 0)

        # print(layer_ind, "controller weight shape", weights.shape, layers_dim[layer_ind])
        # weights = np.tile(weights, reps=(kernel_size, kernel_size, 1, 1)).\
    #                        transpose(1, 2).transpose(0, 3)
                            # transpose(1, 2).transpose(0, 3).transpose(0, 1)
        # weights = np.tile(weights, reps=(kernel_size, kernel_size, 1, 1)).\
        #                        transpose(3, 2, 1, 0)
        return weights

    def get_prev_iter_weights(self, imp_iter):
        run_dir = utils.get_run_dir(self.args)
        model = torch.load(run_dir + str(imp_iter) + '_model.pth.tar')
        # model.eval()
        weights = {}

        idx = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if (idx in self.controller.c_layers):
                    weights[idx] = module[1].weight
                idx += 1
        return weights


def lth(logger, device, args):
    ITERATION = args.exper_imp_total_iter               # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr)
    logger.debug("Warming up the pretrained model")
    max_acc = warm_up(model, train_dl, test_dl, loss_fn, optimizer, args, device)

    similarity = Similarity(args, test_dl, device, run_dir, num_classes)
    act = Activations(model, train_dl, device, args.net_batch_size)
    pruning = Pruner(args, model, act)
    init_state_dict = pruning.init_lth()
    # act.compute_correlations()

    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            # non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr,
            #                             weight_decay=args.net_weight_decay)
            # act.compute_correlations()

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.count_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        for train_iter in range(args.net_train_epochs):

            # Training
            logger.debug(f"Training iteration {train_iter} / {args.net_train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, pruning.mask,
                              args.net_train_per_epoch, device)

            # act.compute_correlations()

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)

            # apply the controller at specific epochs and iteration
            # if ((args.control_on == 1) and
            #     (train_iter == controller.c_epoch) and
            #    (imp_iter in controller.c_iter)):
            #     act.compute_correlations()
            #     corr = act.get_correlations()
            #     pruning.control(corr, act.layers_dim, imp_iter)

            pruning.all_acc[imp_iter, train_iter] = accuracy

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity

    output = [pruning.all_acc,
              similarity.get_similarity(),
              act.get_conns(),
              act.get_gradient(),
              pruning.comp_level]

    return output


def sap(logger, device, args):
    return lth(logger, device, args)


def test_lth(logger, device, args):
    ITERATION = args.exper_imp_total_iter               # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr,
                                weight_decay=args.net_weight_decay)
    # warm up the pretrained model
    max_acc = warm_up(model, train_dl, test_dl, loss_fn, optimizer, args, device)

    similarity = Similarity(args, test_dl, device, run_dir, num_classes)
    act = Activations(model, train_dl, device, args.net_batch_size)
    pruning = Pruner(args, model, act)
    init_state_dict = pruning.init_lth()
    act.compute_correlations()
    act.gradient_flow()

    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, cuz we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            act.compute_correlations()
            act.gradient_flow()
            similarity.cosine_similarity(model, imp_iter)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.count_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training loop
        for train_iter in range(args.net_train_epochs):

            # Training
            logger.debug(f"Training iteration {train_iter} / {args.net_train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, pruning.mask,
                              args.net_train_per_epoch, device)
            act.compute_correlations()
            act.gradient_flow()
            similarity.cosine_similarity(model, imp_iter)

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)

            # apply the controller after some epochs and some iterations
            if ((args.control_on == 1) and
                (train_iter == controller.c_epoch) and
               (imp_iter in controller.c_iter)):
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)

            pruning.all_acc[imp_iter, train_iter] = accuracy

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # act.compute_correlations()
        # logger.debug(f"similarities: {similarity.get_similarity()}")

    output = [pruning.all_acc,
              similarity.get_similarity(),
              act.get_conns(),
              act.get_gradient(),
              pruning.comp_level]

    return output


def ciap(logger, device, args):
    ITERATION = args.exper_imp_total_iter
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr,
                                weight_decay=args.net_weight_decay)
    # warm up the pretrained model
    logger.debug("Pretraining warm-up")
    max_acc = warm_up(model, train_dl, test_dl, loss_fn, optimizer, args, device)

    similarity = Similarity(args, test_dl, device, run_dir, num_classes)
    act = Activations(model, train_dl, device, args.net_batch_size)
    pruning = Pruner(args, model, act)
    init_state_dict = pruning.init_lth()
    act.compute_correlations()
    # act.gradient_flow()

    for imp_iter in tqdm(range(ITERATION)):
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            act.compute_correlations()
            # act.gradient_flow()
            # similarity.cosine_similarity(model, imp_iter)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.count_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training loop
        # for train_iter in range(args.net_train_epochs):
        train_iter, accuracy = 0, 0
        while (train_iter < args.net_train_epochs and accuracy < max_acc):
            # Training
            logger.debug(f"Training iteration {train_iter} / {args.net_train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, pruning.mask,
                              args.net_train_per_epoch, device)
            act.compute_correlations()
            # act.gradient_flow()
            # similarity.cosine_similarity(model, imp_iter)

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)
            pruning.all_acc[imp_iter, train_iter] = accuracy
            if (network.trained_enough(correlation=act.get_correlations(),
                                       epsilon=args.prune_epsilon)):
                break
            train_iter += 1

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # act.compute_correlations()
        # logger.debug(f"similarities: {similarity.get_similarity()}")

    output = [pruning.all_acc,
              similarity.get_similarity(),
              act.get_conns(),
              act.get_gradient(),
              pruning.comp_level]

    return output


def giap(logger, device, args):
    ITERATION = args.exper_imp_total_iter
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr,
                                weight_decay=args.net_weight_decay)
    # warm up the pretrained model
    logger.debug("Pretraining warm-up")
    max_acc = warm_up(model, train_dl, test_dl, loss_fn, optimizer, args, device)

    similarity = Similarity(args, test_dl, device, run_dir, num_classes)
    act = Activations(model, train_dl, device, args.net_batch_size)
    pruning = Pruner(args, model, act)
    init_state_dict = pruning.init_lth()
    # act.compute_correlations()
    act.gradient_flow()

    for imp_iter in tqdm(range(ITERATION)):
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            # act.compute_correlations()
            act.gradient_flow()
            # similarity.cosine_similarity(model, imp_iter)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.count_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training loop
        # for train_iter in range(args.net_train_epochs):
        train_iter, accuracy = 0, 0
        while (train_iter < args.net_train_epochs and accuracy < max_acc):
            # Training
            logger.debug(f"Training iteration {train_iter} / {args.net_train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, pruning.mask,
                              args.net_train_per_epoch, device)
            # act.compute_correlations()
            act.gradient_flow()
            # similarity.cosine_similarity(model, imp_iter)

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)
            pruning.all_acc[imp_iter, train_iter] = accuracy
            if (network.trained_enough(grads=act.get_gradient(),
                                       epsilon=args.prune_epsilon)):
                break
            train_iter += 1

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # act.compute_correlations()
        # logger.debug(f"similarities: {similarity.get_similarity()}")

    output = [pruning.all_acc,
              similarity.get_similarity(),
              act.get_conns(),
              act.get_gradient(),
              pruning.comp_level]

    return output


def experiment(logger, args, device, run_dir):
    logger.debug(f"####### In {args.prune_method} experiment #######")
    acc_list = []
    conns = []
    comp_list = []
    similarity = []
    grads = []

    for i in range(args.exper_num_trial):
        logger.debug(f"In experiment {i} / {args.exper_num_trial}")
        acc, sim, conn, grad, comp = eval(args.prune_method)(logger, device, args)
        acc_list.append(acc)
        comp_list.append(comp)
        similarity.append(sim)
        conns.append(conn)
        grads.append(grad)

        utils.save_vars(save_dir=run_dir+str(i)+"_", similarity=sim,
                        accuracy=acc, conn=conn,
                        grad=grad, comp_level=comp)

    utils.save_vars(save_dir=run_dir, similarity=similarity,
                    accuracies=acc_list,
                    conns=conns,
                    grads=grads,
                    comp_levels=comp_list)

    # ptool.plot_similarity(run_dir, acc_list, comp_list, conn_list)

# def effic_exper(logger, args, device, run_dir):
#     logger.debug("####### In efficiency experiemnt #######")
#     controller = Controller(args)
#     similarity = []
#     acc_list = []
#     grads = []
#     conns = []
#     comp_list = []

#     for i in range(args.exper_num_trial):
#         logger.debug(f"In experiment {i} / {args.exper_num_trial}")
#         all_acc, sim, conn, grad, comp = effic_lth(logger, device, args, controller)
#         acc_list.append(all_acc)
#         similarity.append(sim)
#         conns.append(conn)
#         grads.append(grad)
#         comp_list.append(comp)
#         utils.save_vars(save_dir=run_dir+str(i)+"_", similarity=sim,
#                         accuracies=all_acc, conn=conn,
#                         grad=grad, comp_level=comp)

#     utils.save_vars(save_dir=run_dir, similarity=similarity,
#                     accuracies=acc_list,
#                     conns=conns,
#                     grads=grads,
#                     comp_levels=comp_list)


# def test_exper(logger, args, device, run_dir):
#     logger.debug("####### In efficiency experiemnt #######")
#     controller = Controller(args)
#     acc_list = []
#     similarity = []
#     conns = []
#     grads = []
#     comp_list = []

#     for i in range(args.exper_num_trial):
#         logger.debug(f"In experiment {i} / {args.exper_num_trial}")
#         results = perf_test_lth(logger, device, args, controller)
#         all_acc, sim, conn, grad, comp = results
#         acc_list.append(all_acc)
#         similarity.append(sim)
#         conns.append(conn)
#         grads.append(grad)
#         comp_list.append(comp)
#         utils.save_vars(save_dir=run_dir+str(i)+"_", similarity=sim,
#                         accuracies=all_acc, conn=conn,
#                         grad=grad, comp_level=comp)

#     # Save the variables
#     utils.save_vars(save_dir=run_dir, similarity=similarity,
#                     accuracies=acc_list,
#                     corrs=conns,
#                     grads=grads,
#                     comp_levels=comp_list)

    # Plot
    # ptool.plot_similarity(run_dir, acc_list, comp_list, similarity, corrs, grads)


def main():
    args = utils.get_args()
    logger = utils.setup_logger_dir(args)
    # args = utils.get_yaml_args(args)
    device = utils.get_device(args)
    run_dir = utils.get_run_dir(args)
    experiment(logger, args, device, run_dir)


if __name__ == '__main__':
    main()
