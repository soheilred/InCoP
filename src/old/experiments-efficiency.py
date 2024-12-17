import copy
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle

# Custom Libraries
import utils
from data_loader import Data
from network import Network, train, test
from correlation import Activations
from lth import LTPruning
import constants as C

def main():
    args = utils.get_args()
    logger = utils.setup_logger()
    # preparing the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    ITERATION = 10               # 35 was the default
    num_training_epochs = args.train_epochs // 2     # 100 is the default 
    MODEL_DIR = C.MODEL_ROOT_DIR

    # Copying and Saving Initial State
    network = Network(device, args.arch, pretrained)
    preprocess = network.preprocess
    data = Data(args.batch_size, C.DATA_DIR, dataset, transform=preprocess)
    train_dataloader, test_dataloader = data.train_dataloader, data.test_dataloader
    model = network.set_model()

    pruning = LTPruning(model, args.arch, args.prune_perc_per_layer, train_dataloader, test_dataloader)

    # Weight Initialization
    model.apply(pruning.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.save_model(model, MODEL_DIR, "/initial_state_model.pth.tar")

    # Making Initial Mask
    pruning.init_mask()

    # Optimizer and Loss
    # TODO: why there are two criterion definitions?
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

    # # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    corrs = []
    control_corrs = []
    best_accuracy = 0
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    train_epoch = np.zeros(ITERATION, int)
    all_loss = []
    all_accuracy = []

    # Iterative Magnitude Pruning main loop
    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, cuz we don't prune in the first
        # iteration
        accuracy = -1
        loss_list = []
        acc_list = []
        if imp_iter != 0:
            pruning.prune_once(initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop. Pruning level "
                     f"{comp[imp_iter]}")

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        comp[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        # for train_iter in tqdm(range(num_training_epochs), leave=False):
        while accuracy < args.acc_thrd:
            # avoid overfitting on training data
            logger.debug(f"Iteration {train_epoch[imp_iter]}"
                         f", accuracy:{accuracy:.2f}")
            if len(acc_list) > 4:
                logger.debug(f"last 3: {np.mean(acc_list[-3:]):.2f}")
            if (train_epoch[imp_iter] > 40) and (train_epoch[imp_iter] % 2 == 0):
                if abs(accuracy - np.mean(acc_list[-3:])) < 2:
                    break
            train_epoch[imp_iter] += 1

            # Training
            logger.debug(f"Training iteration {train_epoch[imp_iter]}")
            acc, loss = train(model, train_dataloader, criterion, optimizer,
                              epochs=args.train_epochs, device=device)

            # Test and save the most accurate model
            # logger.debug("Testing...")
            accuracy = test(model, test_dataloader, criterion, device)

            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.debug(f"Best accuracy is updated to {best_accuracy:.2f}")

            # apply the controller after some epochs
            if (train_epoch[imp_iter] == control_at_epoch) and (imp_iter == control_at_iter):
                # network.trained_enough(accuracy, train_dataloader, criterion,
                #                        optimizer, num_epochs, device)
                activations = Activations(model, train_dataloader, device, args.batch_size)
                # control_corrs.append(activations.get_connectivity())
                control_corrs.append(activations.get_correlations())
                # pruning.apply_controller(control_corrs, [2])
                pruning.controller(control_corrs, activations.layers_dim, control_type, imp_iter, [2])
                # pruning.apply_controller(3, control_corrs)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

            loss_list.append(loss)
            acc_list.append(accuracy)

        # Calculate the connectivity
        # network.trained_enough(accuracy, train_dataloader, criterion, optimizer,
        #                        num_epochs, device) 
        activations = Activations(model, train_dataloader, device, args.batch_size)
        corr = activations.get_connectivity()
        corrs.append(corr)
        if imp_iter <= control_at_iter:
            control_corrs.append(activations.get_correlations())

        all_loss.append(loss_list)
        all_accuracy.append(acc_list)
        # save the best model
        bestacc[imp_iter] = best_accuracy

        # Dump Plot values
        # all_loss.dump(C.RUN_DIR + f"{prune_type}_all_loss_{comp_level}.dat")
        # all_accuracy.dump(C.RUN_DIR + f"{prune_type}_all_accuracy_{comp_level}.dat")
        pickle.dump(corrs, open(C.RUN_DIR + args.arch + "_correlation.pkl", "wb"))
        pickle.dump(all_accuracy, open(C.RUN_DIR + args.arch + "_all_accuracy.pkl", "wb"))
        pickle.dump(all_loss, open(C.RUN_DIR + args.arch + "_all_loss.pkl", "wb"))
        pickle.dump(train_epoch, open(C.RUN_DIR + args.arch + "_train_epochs.pkl", "wb"))

        utils.save_model(model, MODEL_DIR, f"{imp_iter + 1}_model.pth.tar")

        # Dumping mask
        # with open(C.RUN_DIR + f"{prune_type}_mask_{comp_level}.pkl", 'wb') as fp:
        #     pickle.dump(pruning.mask, fp)

    # Dumping Values for Plotting
    pickle.dump(comp, open(C.RUN_DIR + args.arch + "_compression.pkl", "wb"))
    # bestacc.dump(C.RUN_DIR + f"{prune_type}_bestaccuracy.dat")
    pickle.dump(bestacc, open(C.RUN_DIR + args.arch + "_best_accuracy.pkl", "wb"))
    con_stability = utils.get_stability(corrs)
    print(con_stability)
    pickle.dump(con_stability,
                open(C.RUN_DIR + args.arch + "_connectivity_stability.pkl", "wb"))
    perform_stability = utils.get_stability(all_loss)
    print(perform_stability)
    pickle.dump(perform_stability,
                open(C.RUN_DIR + args.arch + "_performance_stability.pkl", "wb"))
    utils.plot_experiment(bestacc, corrs, C.RUN_DIR + args.arch +
                          "correlation")
    utils.plot_experiment(bestacc, con_stability, C.RUN_DIR + args.arch +
                          "connection-stability")

if __name__ == '__main__':
    main()
