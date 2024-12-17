import os
import sys
import datetime
import torch
import pickle
import numpy as np
import pathlib
import logging
import logging.config
from torch import nn
from models import ResidualBlock
from torchvision.models.resnet import Bottleneck, BasicBlock

import utils
import plot_tool
from data_loader import Data
from network import Network, train, test
import constants as C
torch.set_printoptions(precision=6)

log = logging.getLogger("sampleLogger")
log.debug("In " + os.uname()[1])


class Activations:
    def __init__(self, model, dataloader, device, batch_size):
        """
        Parameters
        ----------
        model: Network
            The network we want to calculate the connectivity for.
        dataloader: Data
            The data we are drawing from, usuaully the test set.
        device: string
            "cpu" or "cuda"
        batch_size: int
            Batch size
        """
        self.activation = {}
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.batch_size = batch_size
        self.layers_dim = None
        self.layers_idx = None
        self.act_keys = None
        self.hook_handles = []
        self.correlations = []
        self.grads = []
        self.set_layers_idx()
        self.set_act_keys()

    def hook_fn(self, m, i, o):
        """Assign the activations/mean of activations to a matrix
        Parameters
        ----------
        self: type
            description
        m: str
            Layer's name
        i: type
            description
        o: torch tensor
            the activation function output
        """

        tmp = o.detach()
        if len(tmp.shape) > 2:
            self.activation[m] = torch.mean(tmp, axis=(2, 3)).detach().cpu()
            # activation[m] = tmp
        else:
            self.activation[m] = tmp.detach().cpu()

    def hook_layer_idx(self, item_key, hook_handles):
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if (module_idx == item_key):
                    hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                    self.layers_dim.append(module[1].weight.shape)

    def get_parent_child_pairs(self, net, parent, child, pc_dict):
        for name, module in net.named_children():
            if isinstance(module, nn.Sequential) or \
               isinstance(module, BasicBlock) or \
               isinstance(module, Bottleneck):

                print('outer', name, module)
                self.get_parent_child_pairs(module)

            elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print('inner', name, module)
                if len(pc_dict) == 0:
                    parent = name
                    pc_dict[parent] = []
                else:
                    pc_dict[parent].append(child)
                    parent = child
                self.hook_handles.append(module.register_forward_hook(self.hook_fn))

    def hook_all_layers(self, layers_dim, hook_handles):
        """ Hook a handle to all layers that are interesting to us, such as
        Linear or Conv2d.
        Parameters
        ----------
        net: Network
            The network we're looking at
        layers_dim: list
            List of layers that are registered for saving activations
        hook_handles: type
            description

        If it is a sequential, don't register a hook on it but recursively
        register hook on all it's module children
        """
        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
               isinstance(module[1], nn.Linear):
                if "downsample" in module[0]:
                    continue
                hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                layers_dim.append(module[1].weight.shape)

        # Recursive version
        # for name, layer in net._modules.items():
        #     if (isinstance(layer, nn.Sequential) or
        #             isinstance(layer, ResidualBlock) or
        #             isinstance(layer, Bottleneck)):
        #         self.get_all_layers(layer, layers_dim, hook_handles)
        #     elif (isinstance(layer, nn.Conv2d) or
        #           (isinstance(layer, nn.Linear))):
        #         # it's a non sequential. Register a hook
        #         hook_handles.append(layer.register_forward_hook(self.hook_fn))
        #         layers_dim.append(layer.weight.shape)

    def set_layers_idx(self):
        """Find all convolutional and linear layers and add them to layers_ind.
        """
        layers_idx = []
        layers_dim = []
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if "downsample" in module[0]:
                    continue
                layers_idx.append((module_idx, module[0]))
                layers_dim.append(module[1].weight.shape)

        self.layers_dim = layers_dim
        self.layers_idx = layers_idx

    def get_layers_idx(self):
        if self.layers_idx is None:
            self.set_layers_idx()
        return self.layers_idx

    def set_act_keys(self):
        # self.get_parent_child_pairs(self.model)
        layers_dim = []
        hook_handles = []

        self.hook_all_layers(layers_dim, hook_handles)

        with torch.no_grad():
            X, y = next(iter(self.dataloader))
            X, y = X.to(self.device), y.to(self.device)
            self.model(X)
            act_keys = list(self.activation.keys())

        self.act_keys = act_keys
        self.hook_handles = hook_handles

    def get_act_keys(self):
        if self.act_keys is None:
            self.set_act_keys()
        return self.act_keys

    def get_corrs(self):
        """ Compute the individual correlation
        Returns
        -------
        List of 2d tensors, each representing the connectivity between two
        consecutive layer.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)

        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        corrs = []

        for idx in range(num_layers - 5):
            logging.debug(f"working on layer {layers_idx[idx]} {str(act_keys[idx])[:18]}...")
            # prepare an array with the right dimension
            parent_arr = []
            child_arr = []

            with torch.no_grad():
                for batch, (X, y) in enumerate(self.dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    self.model(X)
                    parent_arr.append(self.activation[act_keys[idx]].
                                      detach().cpu().numpy())
                    child_arr.append(self.activation[act_keys[idx + 1]].
                                     detach().cpu().numpy())

            del self.activation[act_keys[idx]]
            self.hook_handles.pop(0)

            parent = np.vstack(parent_arr)
            parent = (parent - parent.mean(axis=0))
            # parent /= np.abs(np.max(parent))
            parent /= parent.std(axis=0)
            child = np.vstack(child_arr)
            child = (child - child.mean(axis=0))
            # child /= np.abs(np.max(child)) # child.std(axis=0)
            child /= child.std(axis=0)
            if np.any(np.isnan(parent)):
                print("nan in layer {layers_idx[idx]}")

            if np.any(np.isnan(child)):
                print("nan in layer {layers_idx[idx + 1]}")
            # corr = np.corrcoef(parent, child, rowvar=False)
            corr = utils.batch_mul(parent, child)
            logging.debug(f"correlation dimension: {corr.shape}, conn: {np.mean(corr):.6f}")
            corrs.append(corr)

        # print(corrs)
        self.correlations.append(corrs)

    def get_conns(self):
        corrs = self.correlations
        conns = []
        for corr in corrs:
            conns.append([elem.mean().item() for elem in corr])
        return conns

    def get_act_layer(self, layers_dim, hook_handles):
        """ Hook a handle to all layers that are interesting to us, such as
        Linear or Conv2d.
        Parameters
        ----------
        net: Network
            The network we're looking at
        layers_dim: list
            List of layers that are registered for saving activations
        hook_handles: type
            description

        If it is a sequential, don't register a hook on it but recursively
        register hook on all it's module children
        """
        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):

                if "downsample" in module[0]:
                    continue

                hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                self.layers_dim.append(module[1].weight.shape)

        # Recursive version
        # for name, layer in net._modules.items():
        #     if (isinstance(layer, nn.Sequential) or
        #             isinstance(layer, ResidualBlock) or
        #             isinstance(layer, Bottleneck)):
        #         self.get_all_layers(layer, layers_dim, hook_handles)
        #     elif (isinstance(layer, nn.Conv2d) or
        #           (isinstance(layer, nn.Linear))):
        #         # it's a non sequential. Register a hook
        #         hook_handles.append(layer.register_forward_hook(self.hook_fn))
        #         layers_dim.append(layer.weight.shape)

    def get_correlations(self):
        return self.correlations

    def compute_correlations(self):
        """ Compute the individual correlation
        Returns
        -------
        List of 2d tensors, each representing the connectivity between two
        consecutive layer.
        """
        log.debug("Compute correlations of activations")
        EPS = 0.000001
        self.model.eval()
        ds_size = len(self.dataloader.dataset)

        layers_idx = self.get_layers_idx()
        # print(layers_idx)
        layers_dim = self.layers_dim
        # print(layers_dim)
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        device = self.activation[act_keys[0]].device

        corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).
                 to(device) for i in range(num_layers - 1)]

        act_means = [torch.zeros(layers_dim[i][0]).to(device)
                     for i in range(num_layers)]
        act_sq_sum = [torch.zeros(layers_dim[i][0]).to(device)
                      for i in range(num_layers)]
        act_max = torch.zeros(num_layers).to(device)

        with torch.no_grad():
            # Compute the mean of activations
            # log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                self.model(X)
                for i in range(num_layers):
                    # summing over all data points in a batch (dim=0)
                    act_means[i] += torch.sum(torch.nan_to_num(
                                              self.activation[act_keys[i]]),
                                              dim=0)
                    act_sq_sum[i] += torch.sum(torch.pow(torch.nan_to_num(
                                               self.activation[act_keys[i]]), 2),
                                               dim=0)
                    act_max[i] = abs(torch.max(act_max[i],
                                     abs(torch.max(self.activation[act_keys[i]]))))

            act_means = [act_means[i] / ds_size for i in range(num_layers)]
            act_sd = [torch.sqrt(act_sq_sum[i] / ds_size -
                                 torch.pow(act_means[i], 2))
                      for i in range(num_layers)]

            # fix maximum activation for layers that are too close to zero
            for i in range(num_layers):
                sign = torch.sign(act_max[i])
                act_max[i] = sign * max(abs(act_max[i]), EPS)
                act_sd[i] = torch.max(act_sd[i], EPS *
                                      torch.ones_like(act_sd[i]).to(device))
                # logging.debug(f"nans in activation sd layer {i}: {torch.isnan(act_sd[i]).any()}")
                # logging.debug(f"nans in activation sd layer {i}: {torch.sum(torch.isnan(act_sd[i].view(-1)))}")
            # log.debug(f"activation mean: {act_means}")
            # log.debug(f"# nans in activation sd: {torch.nonzero(torch.isnan(act_sd.view(-1)))}")
            # log.debug(f"activation max: {act_max}")

            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                #     log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                for i in range(num_layers - 1):
                    f0 = ((self.activation[act_keys[i]] - act_means[i]) /
                          act_sd[i]).T
                    f1 = ((self.activation[act_keys[i + 1]] - act_means[i + 1]) /
                          act_sd[i + 1])
                    corrs[i] += torch.matmul(f0, f1).detach().cpu()

        for i in range(num_layers - 1):
            corrs[i] = corrs[i] / ds_size # (layers_dim[i][0] * layers_dim[i + 1][0])

        self.correlations.append(corrs)

    def get_connectivity(self):
        """Find the connectivity of each layer, the mean of correlation matrix.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)
        num_batch = len(self.dataloader)
        # params = list(self.model.parameters())

        layers_dim = []
        hook_handles = []
        self.get_all_layers(self.model, layers_dim, hook_handles)
        num_layers = len(layers_dim)
        first_run = 1
        torch.set_printoptions(precision=4)

        corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).to(self.device)
                 for i in range(num_layers - 1)]
        activation_means = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]
        sq_sum = [torch.zeros(layers_dim[i][0]).to(self.device)
                         for i in range(num_layers)]
        act_max = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]

        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                if first_run:
                    act_keys = list(self.activation.keys())
                    first_run = 0

                for i in range(num_layers):
                    activation_means[i] += torch.sum(
                        self.activation[act_keys[i]], dim=0)
                    sq_sum[i] += torch.sum(
                        torch.pow(self.activation[act_keys[i]], 2), dim=0)
                    act_max[i] = torch.max(self.activation[act_keys[i]])
            # log.debug("-------------------------------")

            activation_means = [activation_means[i] / ds_size
                                for i in range(num_layers)]
            activation_sd = [torch.pow(sq_sum[i] / ds_size -
                                       torch.pow(activation_means[i], 2), 0.5)
                             for i in range(num_layers)]

            # Compute normalized correlation
            log.debug("Compute normalized correlation")
            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                # if first_run:
                #     act_keys = list(self.activation.keys())
                #     first_run = 0
                for i in range(num_layers - 1):
                    # Normalized activations
                    f0 = torch.div((self.activation[act_keys[i]] -
                                    activation_means[i]), act_max[i]).T
                    f1 = torch.div((self.activation[act_keys[i + 1]] -
                                    activation_means[i + 1]), act_max[i + 1])

                    # Zero-meaned activations
                    # f0 = (self.activation[act_keys[i]] - activation_means[i]).T
                    # f1 = (self.activation[act_keys[i + 1]] - activation_means[i + 1])

                    corrs[i] += torch.matmul(f0, f1)
        # Remove all hook handles
        for handle in hook_handles:
            handle.remove()

        self.model.train()
        return [torch.mean(corrs[i]).item()/(layers_dim[i][0] * layers_dim[i + 1][0])
                for i in range(num_layers - 1)]

    def gradient_flow(self):
        log.debug("Computing the gradient flow")
        self.model.eval()
        grads_mean = []
        grads_norm = []
        # grads = {}

        # for module_idx, module in enumerate(self.model.named_modules()):
        #     if isinstance(module[1], nn.Conv2d) or \
        #                  isinstance(module[1], nn.Linear):
        #         if "downsample" in module[0]:
        #             continue

        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                grads_mean.append(torch.zeros_like(param.grad.cpu().abs().mean()))
                grads_norm.append(torch.zeros_like(param.grad.cpu().abs().mean()))
                # grads_norm.append(param.grad.cpu().abs().norm())
        #         grads.append(module[1].weight.grad.cpu())
                # grads[name] = param.grad.cpu()

        for batch, (X, y) in enumerate(self.dataloader):
            X, y = X.to(self.device), y.to(self.device)
            self.model(X)
            layer_id = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    grads_mean[layer_id] += param.grad.cpu().abs().mean()
                    grads_norm[layer_id] += param.grad.cpu().abs().norm()
                    layer_id += 1
            #         grads.append(module[1].weight.grad.cpu())
                    # grads[name] = param.grad.cpu()

        grads_mean = [grad / batch for grad in grads_mean]
        grads_norm = [grad / batch for grad in grads_norm]
        self.grads.append((grads_mean, grads_norm))
        # self.grads.append(grads)

    def get_gradient(self):
        return self.grads


def main():
    args = utils.get_args()
    logger = utils.setup_logger_dir(args)
    args = utils.get_yaml_args(args)
    device = utils.get_device(args)
    run_dir = utils.get_run_dir(args)
    ITERATION = args.imp_total_iter               # 35 was the default
    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.arch, num_classes, args.pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.debug("Warming up the pretrained model")
    acc, _ = train(model, train_dl, loss_fn, optimizer, None, 1, device)

    act = Activations(model, test_dl, device, args.batch_size)
    corr = act.get_correlations()

    print(corr)
    # plot_tool.plot_connectivity(test_acc, corr)


if __name__ == '__main__':
    main()
