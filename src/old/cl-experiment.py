#!/usr/bin/env python
import utils
from data_loader import Data
from network import Network, train, test
from correlation import Activations
import constants as C
from prune import Pruner
import logging
import logging.config
import torch
import torch.nn as nn


def main():
    device = utils.get_device()
    args = utils.get_args()
    logger = utils.setup_logger()

    # instantiate the model 
    network = Network(device, args.arch)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()

    dataset_list = ["CIFAR10", "MNIST"]

    pruning = Pruner(model, args.prune_perc_per_layer*100, len(dataset_list))

    # Looping over tasks
    for task in range(0, args.task_num):
        # Update the dataset
        data = Data(args.batch_size, C.DATA_DIR, dataset_list[task])
        num_classes = data.get_num_classes()
        train_dl, test_dl = data.train_dataloader, data.test_dataloader

        # prepare the model for the task
        model.add_dataset(str(task), num_classes)
        model.set_dataset(str(task), device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        train_acc, _ = train(model, train_dl, loss_fn, optimizer,
                             args.train_epochs, device)
        pruning.prune_by_percentile()
        pruner.apply_mask()
        pruner.increment_task()



if __name__ == '__main__':
    main()
