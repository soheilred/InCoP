import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle

import argparse
import utils
from josh_prune import SparsePruner
from activation import Activations
import constants as C
from network import Network, train, test
from data_loader import Data

def main():
    logger = utils.setup_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_num", default=1, type=int)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--prune_perc_per_layer", default=10, type=int)
    parser.add_argument("--cores", default=4, type=int)
    parser.add_argument("--freeze_perc", default=16, type=int)
    parser.add_argument("--num_freeze_layers", default=16, type=int)
    parser.add_argument("--freeze_order", default=16, type=int)
    parser.add_argument("--train_biases", default=16, type=int)
    parser.add_argument("--train_bn", default=16, type=int)

    args = parser.parse_args()

    conns = {}
    conn_aves = {}
    composite_mask = None
    all_task_masks = None
    # arch_type = "vgg16"
    arch_type = "alexnet"
    pretrained = True
    dataset = "CIFAR10"
    batch_size = 256
    num_epochs = 1
    num_training_epochs = 10
    lr = 1.2e-3
    MODEL_DIR = C.MODEL_ROOT_DIR + arch_type + "/" + dataset + "/"

    network = Network(device, arch_type, pretrained)
    preprocess = network.preprocess
    data = Data(batch_size, C.DATA_DIR, transform=preprocess)
    train_dataloader, test_dataloader = data.train_dataloader, data.test_dataloader
    model = network.set_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    soheil_corrs = []
    josh_corrs = []
    accs = []

    
    for train_iter in tqdm(range(num_training_epochs)):
        logger.debug(f"Training iteration {train_iter}")
        acc, loss = train(model, train_dataloader, criterion, optimizer,
                          epochs=num_epochs, device=device) 
        accs.append(acc)
        activations = Activations(model, test_dataloader, device, batch_size)
        # soheil_corrs.append(activations.get_correlation())
        soheil_corrs.append(activations.get_activations())

        # prune = SparsePruner(args, model, all_task_masks, composite_mask,
        #                      train_dataloader, test_dataloader, conns,
        #                      conn_aves) 
        # prune.calc_conns()
        # josh_corrs.append(prune.conn_aves)

        # pickle.dump(soheil_corrs, open(C.RUN_DIR + "soheil" + "_correlation.pkl", "wb"))
        # pickle.dump(josh_corrs, open(C.RUN_DIR + "josh" + "_correlation.pkl", "wb"))
        # pickle.dump(accs, open(C.RUN_DIR + "_accuracy.pkl", "wb"))

    print(soheil_corrs)

if __name__ == '__main__':
    main()
