import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.init as init
import pickle
import sys
from collections import OrderedDict

import utils
import plot_tool
from data_loader import Data
from network import Network, train, test
from correlation import Activations
import constants as C
import logging
import logging.config

log = logging.getLogger("sampleLogger")


class Controller:
    def __init__(self, args):
        """Control the IMP's connectivity."""
        self.c_type = args.control_type
        self.c_layers = args.control_layer
        self.c_iter = args.control_iteration
        self.c_epoch = args.control_epoch


class Pruner:
    def __init__(
        self,
        args,
        model,
        train_dataloader=None,
        test_dataloader=None,
        controller=None,
        correlation=None,
    ):
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
        self.mask = None
        self.args = args
        self.prune_perc = args.exper_prune_perc_per_layer * 100
        self.corrs = []
        self.controller = controller
        self.correlation = correlation
        self.num_layers = 0
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.comp_level = np.zeros(args.exper_imp_total_iter, float)
        self.all_acc = np.zeros(
            [args.exper_imp_total_iter, args.net_train_epochs], float
        )
        self.init_state_dict = None
        # self.init_dump()

    def weight_init(self, m):
        """Usage:
        model = Model()
        model.apply(weight_init)
        """
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
            if "weight" in name:
                count = count + 1
        return count

    def init_mask(self):
        """Make an empty mask of the same size as the model."""
        self.num_layers = self.count_layers()
        self.mask = [None] * self.num_layers
        layer_id = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() > 1:
                # tensor = param.data.cpu().numpy()
                # self.mask[layer_id] = np.ones_like(tensor)
                self.mask[layer_id] = param.new_ones(param.size(), dtype=torch.bool)
                # self.mask[layer_id] = torch.ones_like(param.data)
                layer_id += 1

    def reset_weights_to_init(self, initial_state_dict):
        """Reset the remaining weights in the network to the initial values."""
        step = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() > 1:
                weight_dev = param.device
                param.data = (self.mask[step] * initial_state_dict[name]).to(weight_dev)
                step += 1

            if "bias" in name:
                param.data = initial_state_dict[name]

    def prune_by_correlation(self, correlation):
        # Calculate percentile value
        layer_id = 0
        for name, param in self.model.named_parameters():
            # We do not prune bias term
            if "weight" in name:
                tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(correlation), self.prune_perc)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(
                    abs(correlation[layer_id]) < percentile_value,
                    0,
                    self.mask[layer_id],
                )

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[layer_id] = new_mask
                layer_id += 1

    def prune_by_percentile(self):
        # Calculate percentile value
        layer_id = 0
        for name, param in self.model.named_parameters():
            # We do not prune bias term
            if "weight" in name and param.dim() > 1:
                tensor = param.data
                alive = tensor[
                    tensor.nonzero(as_tuple=True)
                ]  # flattened array of nonzero values
                # percentile_value = torch.percentile(alive, self.prune_perc)
                percentile_value = torch.quantile(
                    alive.abs(), self.prune_perc / 100.0
                ).item()

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = torch.where(
                    tensor.abs() < percentile_value, 0, self.mask[layer_id]
                )
                new_mask = new_mask.type(torch.bool).to(weight_dev)

                # tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                # percentile_value = np.percentile(abs(alive), self.prune_perc)

                # # Convert Tensors to numpy and calculate
                # weight_dev = param.device
                # new_mask = np.where(abs(tensor) < percentile_value, 0,
                #                     self.mask[layer_id])

                # Apply new weight and mask
                param.data = tensor * new_mask
                # param.grad *= new_mask
                self.mask[layer_id] = new_mask
                layer_id += 1

    def prune_by_sap(self):
        layer_id = 0
        pivot_param = []
        pivot_mask = []

        for name, param in self.model.named_parameters():
            parameter_type = name.split(".")[-1]
            if "weight" in parameter_type and param.dim() > 1:
                # mask_i = mask.state_dict()[name]
                pivot_param_i = param[self.mask[layer_id]].abs()
                pivot_param.append(pivot_param_i.view(-1))
                pivot_mask.append(self.mask[layer_id].view(-1))
                layer_id += 1

        pivot_param = torch.cat(pivot_param, dim=0).data.abs()
        pivot_mask = torch.cat(pivot_mask, dim=0)
        # p, q, eta_m, gamma = self.prune_mode[1:] # TODO
        p, q, eta_m, gamma = float(0.5), float(1.0), float(0.0), float(1.0)
        beta = 0.9
        sparsity_index = {
            "p": torch.arange(0.1, 1.1, 0.1),
            "q": torch.arange(1.0, 2.1, 0.1),
        }
        p_idx = (sparsity_index["p"] == p).nonzero().item()
        q_idx = (sparsity_index["q"] == q).nonzero().item()
        mask_i = pivot_mask
        si = self.make_si_(self.model, self.mask, sparsity_index)
        si_i = si[p_idx, q_idx]
        d = mask_i.float().sum().to("cpu")
        m = d * (1 + eta_m) ** (q / (p - q)) * (1 - si_i) ** ((q * p) / (q - p))
        m = torch.ceil(m).long()
        retain_ratio = m / d
        prune_ratio = torch.clamp(gamma * (1 - retain_ratio), 0, beta)
        num_prune = torch.floor(d * prune_ratio).long()
        pivot_value = torch.sort(pivot_param.view(-1))[0][num_prune]

        # new_mask = [None] * self.num_layers
        layer_id = 0
        # new_mask = OrderedDict()
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() > 1:
                # tensor = param.data.cpu().numpy()
                # alive = tensor[np.nonzero(tensor)]
                # percentile_value = np.percentile(abs(alive), self.prune_perc)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                # new_mask = np.where(abs(tensor) < percentile_value, 0,
                #                     self.mask[layer_id])

                pivot_mask = (param.data.abs() < pivot_value).to(weight_dev)
                new_mask = torch.where(pivot_mask, False, self.mask[layer_id])

                # Apply new weight and mask
                # param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                param.data = torch.where(
                    new_mask.to(param.device),
                    param.data,
                    torch.tensor(0, dtype=torch.float, device=param.device),
                )

                param.grad.mul_(new_mask)
                self.mask[layer_id] = new_mask
                layer_id += 1

    def make_si_(self, model, mask, si_dict):
        sparsity_index = OrderedDict()
        param_all = []
        mask_all = []
        layer_id = 0
        for name, param in model.state_dict().items():
            parameter_type = name.split(".")[-1]
            if "weight" in parameter_type and param.dim() > 1:
                param_all.append(param.view(-1))
                mask_all.append(self.mask[layer_id].view(-1))
                layer_id += 1
        param_all = torch.cat(param_all, dim=0)
        mask_all = torch.cat(mask_all, dim=0)
        sparsity_index_i = []
        for i in range(len(si_dict["p"])):
            for j in range(len(si_dict["q"])):
                sparsity_index_i.append(
                    self.make_si(
                        param_all, mask_all, -1, si_dict["p"][i], si_dict["q"][j]
                    )
                )
        sparsity_index_i = torch.tensor(sparsity_index_i)
        sparsity_index = sparsity_index_i.reshape(
            (len(si_dict["p"]), len(si_dict["q"]), -1)
        )

        return sparsity_index

    def make_si(self, x, mask, dim, p, q):
        d = mask.to(x.device).float().sum(dim=dim)
        x = x * mask.to(x.device).float()
        si = 1 - (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p) / (
            torch.linalg.norm(x, q, dim=dim).pow(q) / d
        ).pow(1 / q)
        si[d == 0] = 0
        si[si == -float("inf")] = 0
        si[torch.logical_and(si > -1e-5, si < 0)] = 0
        return si

    def prune_once(self, initial_state_dict, correlation=None):
        # if correlation is not None:
        #     self.prune_by_correlation(correlation)
        # else:
        # self.prune_by_percentile()
        self.prune_by_sap()
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
                        mask = mask.logical_or(
                            self.all_task_masks[i][0][module_idx].cuda()
                        )
                weight[mask.eq(0)] = 0.0

        self.model.eval()

    def control(self, corr, layers_dim, imp_iter):
        control_corrs = self.corrs + [corr]
        log.debug(f"apply controller at layer {self.controller.c_layers}")

        # print([corr.shape for corr in control_corrs[-1]])

        # get the weights from previous iteration
        prev_iter_weights = self.get_prev_iter_weights(imp_iter)

        # get connectivity
        connectivity = [
            (
                torch.mean(control_corrs[imp_iter - 1][i])
                / (layers_dim[i][0] * layers_dim[i + 1][0])
            )
            for i in range(len(layers_dim) - 1)
        ]

        # get the coefficient based on connectivity
        for ind in self.controller.c_layers:
            prev_corr = self.correlation_to_weights(
                control_corrs, layers_dim, imp_iter, ind
            )
            prev_weight = prev_iter_weights[ind]

            # type 1
            if self.controller.c_type == 1:
                control_weights = abs(prev_corr)  # / max(connectivity)

            # type 2
            elif self.controller.c_type == 2:
                control_weights = torch.mul(prev_corr, prev_weight)

            # type 3
            elif self.controller.c_type == 3:
                control_weights = abs(connectivity[ind]) / max(
                    connectivity
                )  # * prev_weight

            # type 4
            elif self.controller.c_type == 4:
                control_weights = abs(prev_corr)
                control_weights = (
                    torch.exp(control_weights) / np.exp(control_weights).sum()
                )

            elif self.controller.c_type == 5:
                control_weights = torch.exp(abs(prev_corr))

            self.apply_controller(control_weights, ind)

    def apply_controller(self, control_weights, layer_idx):
        idx = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear):
                if "downsample" in module[0]:
                    continue

                if idx == layer_idx:
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

    def get_cosine_similarity(self, act, base_act, device):
        """Compute the cosine similarity between the optimal network and the
        prunned network's activity.

        Returns
        -------
        List of 2d tensors, each representing the similarity between the
        activations of two networks.
        """
        act.model.eval()
        ds_size = len(act.dataloader.dataset)

        layers_dim = act.layers_dim
        # print(layers_dim)
        num_layers = len(layers_dim)
        act_keys = act.get_act_keys()
        # device = act.activation[act_keys[0]].device

        # corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).
        #          to(device) for i in range(num_layers - 1)]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = torch.zeros(num_layers)
        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(act.dataloader):
                X, y = X.to(device), y.to(device)
                act.model(X)
                base_act.model(X)

                for i in range(num_layers):
                    f0 = act.activation[act_keys[i]]
                    f1 = base_act.activation[base_act.act_keys[i]]
                    # corrs[i] += torch.matmul(f0, f1).detach().cpu()
                    similarities[i] += torch.mean(cos(f0, f1).detach().cpu())

        for i in range(num_layers - 1):
            similarities[i] = similarities[i] / ds_size

        return similarities

    def correlation_to_weights(self, control_corrs, layers_dim, imp_iter, layer_ind):
        # the + 1 is for matching to the connectivity's dimension
        weights = control_corrs[imp_iter - 1][layer_ind - 1]
        # weights = control_corrs[0][layer_ind - 1]
        kernel_size = layers_dim[layer_ind][-1]
        weights = weights.repeat([kernel_size, kernel_size, 1, 1]).permute(3, 2, 1, 0)

        # print(layer_ind, "controller weight shape", weights.shape, layers_dim[layer_ind])
        # weights = np.tile(weights, reps=(kernel_size, kernel_size, 1, 1)).\
        #                        transpose(1, 2).transpose(0, 3)
        # transpose(1, 2).transpose(0, 3).transpose(0, 1)
        # weights = np.tile(weights, reps=(kernel_size, kernel_size, 1, 1)).\
        #                        transpose(3, 2, 1, 0)
        return weights

    def get_prev_iter_weights(self, imp_iter):
        run_dir = utils.get_run_dir(self.args)
        model = torch.load(run_dir + str(imp_iter) + "_model.pth.tar")
        # model.eval()
        weights = {}

        idx = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear):
                if idx in self.controller.c_layers:
                    weights[idx] = module[1].weight
                idx += 1
        return weights


def perf_lth(logger, device, args, controller):
    ITERATION = args.exper_imp_total_iter  # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr)
    logger.debug("Warming up the pretrained model")
    acc, _ = train(model, train_dl, loss_fn, optimizer, None, args.net_warmup, device)

    pruning = Pruner(args, model, train_dl, test_dl, controller)
    init_state_dict = pruning.init_lth()
    connectivity = []

    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            # non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.net_lr, weight_decay=args.net_weight_decay
            )

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.count_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        for train_iter in range(args.net_train_epochs):
            # Training
            logger.debug(f"Training iteration {train_iter} / {args.net_train_epochs}")
            acc, loss = train(
                model,
                train_dl,
                loss_fn,
                optimizer,
                pruning.mask,
                args.net_train_per_epoch,
                device,
            )

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)

            # apply the controller at specific epochs and iteration
            if (
                (args.control_on == 1)
                and (train_iter == controller.c_epoch)
                and (imp_iter in controller.c_iter)
            ):
                act = Activations(model, train_dl, device, args.net_batch_size)
                # corr_0 = act.get_corrs()
                # corr_1 = act.get_correlations()
                # print([(torch.from_numpy(corr_0[i]) - corr_1[i]).sum() for i in range(len(corr_0))])
                # import ipdb; ipdb.set_trace()
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)

            pruning.all_acc[imp_iter, train_iter] = accuracy

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # if (imp_iter <= controller.c_iter):
        activations = Activations(model, train_dl, device, args.net_batch_size)
        pruning.corrs.append(activations.get_correlations())
        connectivity.append(activations.get_conns(pruning.corrs[imp_iter]))
        # utils.save_vars(corrs=pruning.corrs, all_accuracies=pruning.all_acc)

    return pruning.all_acc, connectivity, pruning.comp_level


def perf_test_lth(logger, device, args, controller):
    ITERATION = args.exper_imp_total_iter  # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.net_lr, weight_decay=args.net_weight_decay
    )
    # warm up the pretrained model
    acc, _ = train(model, train_dl, loss_fn, optimizer, None, args.net_warmup, device)

    pruning = Pruner(args, model, train_dl, test_dl, controller)
    act = Activations(model, train_dl, device, args.net_batch_size)
    init_state_dict = pruning.init_lth()
    # connectivity = []
    # corrs = []
    similarities = []

    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, cuz we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            # corr = act.get_correlations()
            # corrs.append(corr)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.count_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training loop
        for train_iter in range(args.net_train_epochs):
            # Training
            logger.debug(f"Training iteration {train_iter} / {args.net_train_epochs}")
            acc, loss = train(
                model,
                train_dl,
                loss_fn,
                optimizer,
                pruning.mask,
                args.net_train_per_epoch,
                device,
            )
            # corr = act.get_correlations()
            # corrs.append(corr)
            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)

            # apply the controller after some epochs and some iterations

            if (
                (args.control_on == 1)
                and (train_iter == controller.c_epoch)
                and (imp_iter in controller.c_iter)
            ):
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)

            pruning.all_acc[imp_iter, train_iter] = accuracy

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # pruning.corrs.append(act.get_correlations())
        # connectivity.append(act.get_conns(pruning.corrs[imp_iter]))

        # Setting up the base network for activations
        base_network = Network(device, args.net_arch, num_classes, args.net_pretrained)

        base_model = base_network.set_model()
        base_model = utils.load_model(base_model, run_dir, "1_model.pth.tar")
        base_model.eval()
        base_act = Activations(base_model, train_dl, device, args.net_batch_size)
        similarities.append(pruning.get_cosine_similarity(act, base_act, device))
        logger.debug(f"similarities: {similarities}")

    return pruning.all_acc, similarities, pruning.comp_level


def effic_lth(logger, device, args, controller):
    ITERATION = args.net_imp_total_iter
    run_dir = utils.get_run_dir(args)
    data = Data(args.net_batch_size, C.DATA_DIR, args.net_dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.net_arch, num_classes, args.net_pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.net_lr)

    pruning = Pruner(args, model, train_dl, test_dl, controller)
    init_state_dict = pruning.init_lth()
    connectivity = []
    all_acc = []
    train_iter = np.zeros(ITERATION, int)
    max_acc = 1

    for imp_iter in tqdm(range(ITERATION)):
        accuracy = -1
        acc_list = []
        # except for the first iteration, cuz we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.net_lr, weight_decay=1e-4
            )

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        # comp_level = utils.print_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training loop
        while train_iter[imp_iter] < 30:
            if train_iter[imp_iter] > controller.c_epoch:
                if accuracy > args.net_acc_thrd * max_acc / 100.0:
                    break

            # Training
            logger.debug(
                f"Accuracy {accuracy:.2f} at training iteration "
                f"{train_iter[imp_iter]}, thsd: "
                f"{args.net_acc_thrd * max_acc / 100.0}"
            )
            acc, loss = train(
                model, train_dl, loss_fn, optimizer, args.net_train_per_epoch, device
            )

            # Test and save the most accurate model
            logger.debug("Testing...")
            accuracy = test(model, test_dl, loss_fn, device)
            acc_list.append(accuracy)

            # apply the controller after some epochs and some iterations
            if (train_iter[imp_iter] == controller.c_epoch) and (
                imp_iter == controller.c_iter
            ):
                act = Activations(model, test_dl, device, args.net_batch_size)
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=args.net_lr, weight_decay=1e-4
                )

            # increment the training iterator
            train_iter[imp_iter] += 1

        all_acc.append(acc_list)
        logger.debug(all_acc)
        max_acc = max(all_acc[0])

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        activations = Activations(model, test_dl, device, args.net_batch_size)
        pruning.corrs.append(activations.get_correlations())
        connectivity.append(activations.get_conns(pruning.corrs[imp_iter]))
        # utils.save_vars(corrs=pruning.corrs, all_accuracies=pruning.all_acc)

    return all_acc, connectivity


def perf_exper(logger, args, device, run_dir):
    logger.debug("####### In performance experiment #######")
    controller = Controller(args)
    acc_list = []
    conn_list = []
    comp_level_list = []

    for i in range(args.exper_num_trial):
        logger.debug(f"In experiment {i} / {args.exper_num_trial}")
        all_acc, conn, comp = perf_lth(logger, device, args, controller)
        acc_list.append(all_acc)
        conn_list.append(conn)
        comp_level_list.append(comp)
        utils.save_vars(
            save_dir=run_dir + str(i) + "_",
            conn=conn,
            all_accuracies=all_acc,
            comp_level=comp,
        )
        # plot_tool.plot_all_accuracy(all_acc, C.OUTPUT_DIR + str(i) +
        #                             "all_accuracies")

    # all_acc = np.mean(acc_list, axis=0)
    # conn = np.mean(conn_list, axis=0)
    # plot_tool.plot_all_accuracy(all_acc, C.OUTPUT_DIR + "all_accuracies")
    # utils.save_vars(save_dir=run_dir, conn=conn, all_accuracies=all_acc)
    utils.save_vars(
        save_dir=run_dir,
        conn=conn_list,
        all_accuracies=acc_list,
        comp_level=comp_level_list,
    )


def effic_exper(logger, args, device, run_dir):
    logger.debug("####### In efficiency experiemnt #######")
    controller = Controller(args)
    acc_list = []
    conn_list = []

    for i in range(args.exper_num_trial):
        logger.debug(f"In experiment {i} / {args.num_trial}")
        all_acc, conn = effic_lth(logger, device, args, controller)
        acc_list.append(all_acc)
        conn_list.append(conn)
        utils.save_vars(
            save_dir=run_dir + str(i) + "_", conn=conn, all_accuracies=all_acc
        )
        # plot_tool.plot_all_accuracy(all_acc, C.OUTPUT_DIR + str(i) +
        #                             "all_accuracies")

    # all_acc = np.mean(acc_list, axis=0)
    # conn = np.mean(conn_list, axis=0)
    # plot_tool.plot_all_accuracy(all_acc, C.OUTPUT_DIR + "all_accuracies")
    utils.save_vars(save_dir=run_dir, conn=conn_list, all_accuracies=acc_list)


def test_exper(logger, args, device, run_dir):
    logger.debug("####### In efficiency experiemnt #######")
    controller = Controller(args)
    acc_list = []
    similarity = []
    comp_level_list = []

    for i in range(args.exper_num_trial):
        logger.debug(f"In experiment {i} / {args.exper_num_trial}")
        all_acc, sim, comp = perf_connectivity_lth(logger, device, args, controller)
        acc_list.append(all_acc)
        similarity.append(sim)
        comp_level_list.append(comp)
        utils.save_vars(
            save_dir=run_dir + str(i) + "_",
            similarity=sim,
            all_accuracies=all_acc,
            comp_level=comp,
        )

    utils.save_vars(
        save_dir=run_dir,
        similarity=similarity,
        all_accuracies=acc_list,
        comp_level=comp,
    )


def main():
    args = utils.get_args()
    logger = utils.setup_logger_dir(args)
    # args = utils.get_yaml_args(args)
    device = utils.get_device(args)

    run_dir = utils.get_run_dir(args)
    if args.exper_type == "performance":
        perf_exper(logger, args, device, run_dir)

    elif args.exper_type == "efficiency":
        effic_exper(logger, args, device, run_dir)

    if args.exper_type == "test":
        test_exper(logger, args, device, run_dir)

    else:
        sys.exit("Wrong experiment type")


if __name__ == "__main__":
    main()
