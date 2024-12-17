import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle

# Custom Libraries
import utils
from data_loader import Data
from activation import Activations
from lth import LTPruning
import constants as C
import logging
log = logging.getLogger("sampleLogger")

class Controller:
    def __init__(self, model, cntr_layer, cntr_epoch, cntl_iter):
        """Control the activations."""
        self.model = model
        self.control_at_layer = cntr_layer
        self.control_at_epoch = cntr_epoch
        self.control_at_iter = cntl_iter
    
    def get_controller(model, corrs, imp_iter, train_iter):
        if (train_iter == self.control_at_epoch) and (imp_iter == self.control_at_iter):
            coef = corrs[0][self.control_at_layer]
        else:
            coef = 1
        # model_shape = model.controlled_layer.shape
        return coef * torch.ones([128, 112, 112]).to('cuda')

def train(model, dataloader, loss_fn, optimizer, epochs, device, controller):
    log.debug('Training...')
    size = len(dataloader.dataset)

    for t in range(epochs):
        log.debug(f"Epoch {t+1}")
        correct = 0

        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X, control)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                # log.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= size
        log.debug(f"Training Error: Accuracy: {(100*correct):>0.1f}%")
    return correct, loss

def test(model, dataloader, loss_fn, device, control=None):
    log.debug('Testing')
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, control)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
    log.debug(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100. * correct


def main():
    logger = utils.setup_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")
    if torch.cuda.is_available():
        logger.debug("Name of the Cuda Device: " +
                     torch.cuda.get_device_name())

    # setting hyperparameters
    # setting hyperparameters
    # arch_type = "vgg11"
    arch_type = "vgg16"
    # arch_type = "resnet"
    # arch_type = "alexnet"
    pretrained = True
    # pretrained = False
    dataset = "CIFAR10"
    # NOTE First Pruning Iteration is of No Compression
    batch_size = 64
    num_epochs = 2
    ITERATION = 10               # 35 was the default
    num_training_epochs = int(20 / num_epochs)     # 100 is the default 
    control_at_iter = 1
    control_at_epoch = 2
    prune_percent = 10
    lr = 1.2e-3
    logger.debug(f"Directory: {C.RUN_DIR}")
    logger.debug(f"{ITERATION} iterations, {num_training_epochs} epochs "
                 f"controller appliles at iteration {control_at_iter}"
                 f", epoch {control_at_epoch}")

    # MODEL_DIR = C.MODEL_ROOT_DIR + arch_type + "/" + dataset + "/"
    MODEL_DIR = C.MODEL_ROOT_DIR

    # Copying and Saving Initial State
    # network = Network(device, arch_type, pretrained)
    # model = network.set_model()
    data = Data(batch_size, C.DATA_DIR)
    train_dataloader, test_dataloader = data.train_dataloader, data.test_dataloader
    model = VGG16().to(device)

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

        controller = Controller(model, 2, control_at_epoch, control_at_iter)

        # Training the network
        for train_iter in tqdm(range(num_training_epochs), leave=False):
            # Training
            logger.debug(f"Training iteration {train_iter} / {num_training_epochs}")
            control = controller.get_controller(control_corrs, imp_iter, train_iter)
            acc, loss = train(model, train_dataloader, criterion, optimizer,
                              num_epochs, device, control)

            # Test and save the most accurate model
            # logger.debug("Testing...")
            accuracy = test(model, test_dataloader, criterion, device)

            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.save_model(model, MODEL_DIR, f"{imp_iter}_model.pth.tar")

            # apply the controller after some epochs
            if (train_iter == control_at_epoch) and (imp_iter == control_at_iter):
                # network.trained_enough(accuracy, train_dataloader, criterion,
                #                        optimizer, num_epochs, device)
                activations = Activations(model, test_dataloader, device, batch_size)
                # control_corrs.append(activations.get_connectivity())
                control_corrs.append(activations.get_correlations())
                # pruning.apply_controller(control_corrs, [2])
                # pruning.controller(control_corrs, imp_iter, [2])
                # pruning.apply_controller(3, control_corrs)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            all_loss[imp_iter, train_iter] = loss
            all_accuracy[imp_iter, train_iter] = accuracy

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
