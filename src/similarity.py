import utils
import torch
import torch.nn as nn
from network import Network
from activation import Activations
import logging
import logging.config

log = logging.getLogger("sampleLogger")

class Similarity:
    def __init__(self, args, dataloader, device, run_dir, num_classes):
        self.args = args
        self.iteration = 0
        self.dataloader = dataloader
        self.base_model = None
        self.base_act = None
        self.run_dir = run_dir
        self.device = device
        self.num_classes = num_classes
        self.similarities = []

    def set_base_model(self):
        base_network = Network(self.device, self.args.net_arch, self.num_classes,
                               self.args.net_pretrained)
        self.base_model = base_network.set_model()
        self.base_model.eval()
        self.base_model = utils.load_model(self.base_model, self.run_dir, "1_model.pth.tar")
        self.base_act = Activations(self.base_model, self.dataloader,
                                    self.device,
                                    self.args.net_batch_size)
        self.iteration += 1

    def cosine_similarity(self, model, imp_iter):
        """ Compute the cosine similarity between the optimal network and the
        prunned network's activity.

        Returns
        -------
        List of 2d tensors, each representing the similarity between the
        activations of two networks.
        """
        if imp_iter < 1:
            return

        if imp_iter == 1 and self.iteration == 0:
            self.set_base_model()

        log.debug("Computing the cosine similarity")
        act = Activations(model, self.dataloader, self.device,
                          self.args.net_batch_size)
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
            log.debug("Compute similarity of activations")
            for batch, (X, y) in enumerate(act.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                act.model(X)
                self.base_act.model(X)

                for i in range(num_layers):
                    f0 = act.activation[act_keys[i]]
                    f1 = self.base_act.activation[self.base_act.act_keys[i]]
                    # corrs[i] += torch.matmul(f0, f1).detach().cpu()
                    similarities[i] += torch.mean(cos(f0, f1).detach().cpu())

        for i in range(num_layers - 1):
            similarities[i] = similarities[i] / ds_size

        # return similarities
        self.similarities.append(similarities)
        self.iteration += 1

    def get_similarity(self):
        return self.similarities
