import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle

# Custom Libraries
import utils
from data_loader import Data
from network import Network, train, test
from activation import Activations
from lth import LTPruning
import constants as C


def main():
    device = utils.get_device()
    args = utils.get_args()
    logger = utils.setup_logger()

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    # Optimizer and Loss
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    MODEL_DIR = C.MODEL_ROOT_DIR

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    ITERATION = args.imp_total_iter               # 35 was the default

    pruning = LTPruning(model, args.arch, args.prune_perc_per_layer*100,
                        train_dl, test_dl)
    init_state_dict = pruning.initialize_lth()
        
    corrs = []
    control_corrs = []

    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_accuracy = np.zeros([ITERATION, args.train_epochs], float)

    # Iterative Magnitude Pruning main loop
    for imp_iter in tqdm(range(ITERATION)):
        best_accuracy = 0
        # except for the first iteration, cuz we don't prune in the first
        # iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop. Pruning level "
                     f"{comp[imp_iter]}")

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        comp[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        for train_iter in tqdm(range(args.train_epochs), leave=False):

            # Training
            logger.debug(f"Training iteration {train_iter} / {args.train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer,
                              epochs=args.train_per_epoch, device=device)

            # Test and save the most accurate model
            # logger.debug("Testing...")
            accuracy = test(model, test_dl, loss_fn, device)

            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.save_model(model, MODEL_DIR,
                                 f"{imp_iter + 1}_model.pth.tar")

            # apply the controller after some epochs
            if (train_iter == args.control_at_epoch) and \
                (imp_iter == args.control_at_iter):
                # network.trained_enough(accuracy, train_dataloader, loss_fn,
                #                        optimizer, num_epochs, device)
                activations = Activations(model, test_dl, device, args.batch_size)
                control_corrs.append(activations.get_correlations())
                pruning.controller(control_corrs, activations.layers_dim,
                                   args.control_type, imp_iter, [2])
                # pruning.apply_controller(3, control_corrs)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                             weight_decay=1e-4)


            all_accuracy[imp_iter, train_iter] = accuracy


        # Calculate the connectivity
        activations = Activations(model, test_dl, device, args.batch_size)
        corr = activations.get_connectivity()
        corrs.append(corr)
        if imp_iter <= args.control_at_iter:
            control_corrs.append(activations.get_correlations())

        # save the best model
        # bestacc[imp_iter] = best_accuracy

        utils.save_vars(corrs=corrs, all_accuracy=all_accuracy)
        # Dumping mask

    # Dumping Values for Plotting
    utils.save_vars(corrs=corrs, all_accuracy=all_accuracy)
    # utils.save_vars(comp=comp, bestacc=bestacc)
    con_stability = utils.get_stability(corrs)
    print(con_stability)

if __name__ == '__main__':
    main()
