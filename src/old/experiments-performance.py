import copy
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
from correlation import Activations
from prune import LTPruning
import constants as C


def main():
    logger = utils.setup_logger()
    # preparing the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")
    if torch.cuda.is_available():
        logger.debug("Name of the Cuda Device: " +
                     torch.cuda.get_device_name())

    # setting hyperparameters
    # arch_type = "vgg11"
    # arch_type = "vgg16"
    arch_type = "resnet"
    # arch_type = "alexnet"
    pretrained = True
    # pretrained = False
    dataset = "CIFAR10"
    # dataset = "MNIST"

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    batch_size = 64
    num_epochs = 2
    ITERATION = 10               # 35 was the default
    num_training_epochs = int(20 / num_epochs)     # 100 is the default 
    control_at_iter = 1
    control_at_epoch = 2
    control_type = 1
    prune_percent = 10
    prune_type = "lt"
    lr = 1.2e-3
    logger.debug(f"In file {sys.argv[0]}")
    logger.debug(f"Directory: {C.RUN_DIR}")
    logger.debug(f"Architecture: {arch_type}, Data: {dataset}, Controller: {control_type}")
    logger.debug(f"{ITERATION} iterations, {num_training_epochs} epochs "
                 f"controller appliles at iteration {control_at_iter}"
                 f", epoch {control_at_epoch}")

    # MODEL_DIR = C.MODEL_ROOT_DIR + arch_type + "/" + dataset + "/"
    MODEL_DIR = C.MODEL_ROOT_DIR

    # Copying and Saving Initial State
    network = Network(device, arch_type, pretrained)
    preprocess = network.preprocess
    data = Data(batch_size, C.DATA_DIR, dataset, transform=preprocess)
    train_dataloader, test_dataloader = data.train_dataloader, data.test_dataloader
    model = network.set_model()

    corrs = []
    control_corrs = []

    pruning = LTPruning(model, arch_type, prune_percent, train_dataloader, test_dataloader)

    # Weight Initialization
    model.apply(pruning.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.save_model(model, MODEL_DIR, "/initial_state_dict.pth.tar")

    # Making Initial Mask
    pruning.init_mask()

    # Optimizer and Loss
    # TODO: why there are two criterion definitions?
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)


    # # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    best_accuracy = 0
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_loss = np.zeros([ITERATION, num_training_epochs], float)
    all_accuracy = np.zeros([ITERATION, num_training_epochs], float)

    # Iterative Magnitude Pruning main loop
    for imp_iter in tqdm(range(ITERATION)):
        best_accuracy = 0
        # except for the first iteration, cuz we don't prune in the first
        # iteration
        if imp_iter != 0:
            pruning.prune_once(initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop. Pruning level "
                     f"{comp[imp_iter]}")

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        comp[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        for train_iter in tqdm(range(num_training_epochs), leave=False):

            # Training
            logger.debug(f"Training iteration {train_iter} / {num_training_epochs}")
            acc, loss = train(model, train_dataloader, criterion, optimizer,
                              epochs=num_epochs, device=device)

            # Test and save the most accurate model
            # logger.debug("Testing...")
            accuracy = test(model, test_dataloader, criterion, device)

            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.save_model(model, MODEL_DIR, f"{imp_iter + 1}_model.pth.tar")

            # apply the controller after some epochs
            if (train_iter == control_at_epoch) and (imp_iter == control_at_iter):
                # network.trained_enough(accuracy, train_dataloader, criterion,
                #                        optimizer, num_epochs, device)
                activations = Activations(model, test_dataloader, device, batch_size)
                # control_corrs.append(activations.get_connectivity())
                control_corrs.append(activations.get_correlations())
                # pruning.apply_controller(control_corrs, [2])
                pruning.controller(control_corrs, activations.layers_dim, control_type, imp_iter, [2])
                # pruning.apply_controller(3, control_corrs)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


            all_loss[imp_iter, train_iter] = loss
            all_accuracy[imp_iter, train_iter] = accuracy


            # Frequency for Printing Accuracy and Loss
            # if train_iter % print_freq == 0:
            #     logger.debug(f'Train Epoch: {train_iter}/{num_training_epochs} \n'
            #                  f'Loss: {loss:.6f} Accuracy: {accuracy:.2f}% \n'
            #                  f'Best Accuracy: {best_accuracy:.2f}%\n')

        # Calculate the connectivity
        # network.trained_enough(accuracy, train_dataloader, criterion, optimizer,
        #                        num_epochs, device) 
        activations = Activations(model, test_dataloader, device, batch_size)
        corr = activations.get_connectivity()
        corrs.append(corr)
        if imp_iter <= control_at_iter:
            control_corrs.append(activations.get_correlations())

        # save the best model
        bestacc[imp_iter] = best_accuracy

        # Dump Plot values
        # all_loss.dump(C.RUN_DIR + f"{prune_type}_all_loss_{comp_level}.dat")
        # all_accuracy.dump(C.RUN_DIR + f"{prune_type}_all_accuracy_{comp_level}.dat")
        pickle.dump(corrs, open(C.RUN_DIR + arch_type + "_correlation.pkl", "wb"))
        pickle.dump(all_accuracy, open(C.RUN_DIR + arch_type + "_all_accuracy.pkl", "wb"))
        pickle.dump(all_loss, open(C.RUN_DIR + arch_type + "_all_loss.pkl", "wb"))

        # Dumping mask
        # with open(C.RUN_DIR + f"{prune_type}_mask_{comp_level}.pkl", 'wb') as fp:
        #     pickle.dump(pruning.mask, fp)

    # Dumping Values for Plotting
    pickle.dump(comp, open(C.RUN_DIR + arch_type + "_compression.pkl", "wb"))
    # bestacc.dump(C.RUN_DIR + f"{prune_type}_bestaccuracy.dat")
    pickle.dump(bestacc, open(C.RUN_DIR + arch_type + "_best_accuracy.pkl", "wb"))
    con_stability = utils.get_stability(corrs)
    print(con_stability)
    pickle.dump(con_stability,
                open(C.RUN_DIR + arch_type + "_connectivity_stability.pkl", "wb"))
    perform_stability = utils.get_stability(all_loss)
    print(perform_stability)
    pickle.dump(perform_stability,
                open(C.RUN_DIR + arch_type + "_performance_stability.pkl", "wb"))
    utils.plot_experiment(bestacc, corrs, C.RUN_DIR + arch_type +
                          "correlation")
    utils.plot_experiment(bestacc, con_stability, C.RUN_DIR + arch_type +
                          "connection-stability")

if __name__ == '__main__':
    main()
