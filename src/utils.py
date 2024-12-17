import os
import torch
import pickle
import argparse
import json
import yaml
import configparser
from torch import nn
import numpy as np
import logging
import constants as C
import matplotlib.pyplot as plt
from pathlib import Path
# from varname import nameof

import logging
import logging.config
logger = logging.getLogger("sampleLogger")


# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, directory, name):
    checkdir(directory)
    torch.save(model.state_dict(), directory + name)

def load_model(model, directory, name):
    checkdir(directory)
    model.load_state_dict(torch.load(directory + name))
    return model

def retrieve_name(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def save_vars(**variables):
    save_dir = ""
    for varname, value in variables.items():
        # print(nameof(var))
        if varname == "save_dir":
            save_dir = value
        else:
            pickle.dump(value, open(save_dir + varname + ".pkl", "wb"))
    
def load_checkpoints(args):
    save_prefix = C.CHPT_DIR + str(args.dataset) +\
                str(args.prune_ratio) + str(args.run_id) +\
                str(args.task_num) 
    previoustaskpath = C.CHPT_DIR + str(args.dataset) +\
                        str(args.prune_ratio) + str(args.run_id) +\
                        str(args.task_num-1) 
    os.makedirs(save_prefix, exist_ok = True)
    os.makedirs(previoustaskpath, exist_ok = True)

    trainedpath = os.path.join(save_prefix, "trained.pt")
    initpath = os.path.join(previoustaskpath, "final.pt")

    if os.path.isfile(initpath) == False and args.task_num == 0:
        print("initializing model",flush = True)
        utils.init_dump(args.arch, initpath)
    
    if os.path.isfile(os.path.join(previoustaskpath,"final.pt")) == True and (args.mode == "t" or args.mode == "all"):
        ckpt = torch.load(os.path.join(previoustaskpath,"final.pt"))

    elif os.path.isfile(os.path.join("../checkpoints/",str(args.dataset),str(args.prune_ratio),str(args.run_id),str((len(taskset)-1)), "final.pt")) == True and args.mode == "e":
        ckpt = torch.load(os.path.join("../checkpoints/", str(args.dataset), str(args.prune_ratio), str(args.run_id), str((len(taskset)-1)), "final.pt"))

    elif os.path.isfile(trainedpath) == True and (args.mode == "p" or args.mode == "c"):
        ckpt = torch.load(trainedpath)

    else:
        print("No checkpoint file found")
        return 0


def get_device(args):
    device = f"cuda:{args.exper_gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")
    if torch.cuda.is_available():
        logger.debug("Name of the Cuda Device: " +
                     torch.cuda.get_device_name())
    return device

def batch_mul(a, b):
    # in_shape = mat.shape[:2]
    # res = torch.stack([v[i, j] * mat[i, j, :, :] for i in range(in_shape[0]) for j in range(in_shape[1])])
    # return res.reshape(mat.shape)
    out = np.mean([np.nan_to_num(np.outer(a[i], b[i])) for i in
                  range(a.shape[0])], axis=0)
    return out


def get_vars(arch, out_dir):
    # train_acc = pickle.load(open("vgg16" + "_training_acc.pkl", "rb"))
    corrs = pickle.load(open(out_dir + arch + "_correlation.pkl", "rb"))
    all_accuracy = pickle.load(open(out_dir + arch + "_all_accuracy.pkl", "rb"))
    # max_accuracy = np.max(all_accuracy, axis=1)
    best_accuracy = pickle.load(open(out_dir + arch + "_best_accuracy.pkl", "rb"))
    all_loss = pickle.load(open(out_dir + arch + "_all_loss.pkl", "rb"))
    # compression = pickle.load(open(out_dir + "vgg16" + "_compression.pkl", "rb"))
    # perf_stability = pickle.load(open(out_dir + "vgg16" + "_performance_stability.pkl", "rb")) 
    # connect_stability = pickle.load(open(out_dir + "vgg16" + "_connectivity_stability.pkl", "rb")) 
    return all_accuracy, corrs

def get_mean_train_epochs(arch, out_dir):
    epochs = []
    for odir in out_dir:
        epochs.append(pickle.load(open(C.OUTPUT_DIR + odir + arch + "_train_epochs.pkl", "rb")))
    print(np.sum(epochs, axis=1))
    return np.mean(epochs, axis=0)

def get_max_accuracy(arch, exper_dirs):
    acc_list = []
    corr_list = []
    for i in range(len(exper_dirs)):
        acc, corr = get_vars(arch, C.OUTPUT_DIR + exper_dirs[i])
        acc_list.append(acc)
        corr_list.append(corr)

    acc_mean = np.mean(np.array(acc_list), axis=0)
    acc_max = np.max(acc_mean, axis=1)
    # corr_mean = np.mean(np.array(corr_list), axis=0)
    return acc_max

def get_mean_accuracy(arch, exper_dirs):
    acc_list = []
    corr_list = []
    for i in range(len(exper_dirs)):
        acc, corr = get_vars(arch, C.OUTPUT_DIR + exper_dirs[i])
        acc_list.append(acc)
        corr_list.append(corr)

    acc_mean = np.mean(np.array(acc_list), axis=0)
    print(np.mean(acc_list, axis=(1, 0)))
    return acc_mean


def count_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.to('cpu')
        nz_count = torch.count_nonzero(tensor)
        total_params = torch.prod(torch.tensor(tensor.shape))
        nonzero += nz_count
        total += total_params
        # print(f'{name:10} | nz = {nz_count:4} / {total_params:5} ({100 * nz_count / total_params:6.2f}%) | pruned = {total_params - nz_count :4} | shape = {tensor.shape}')
    #     print(f'{name[:30]: <30} | ({100 * nz_count / total_params:5.1f}%) | pruned = {total_params - nz_count :4} | dim = {tensor.shape}')
    # print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    # # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    return round((nonzero / total).item() * 100, 1)

def setup_logger_dir(args):
    Path(C.RUN_DIR).mkdir(parents=True, exist_ok=True)
    run_dir = get_run_dir(args)
    LOG_FILENAME = run_dir + 'out.log'
    logging.config.fileConfig(C.LOG_CONFIG_DIR,
                              defaults={'logfilename': LOG_FILENAME})
    logger = logging.getLogger("sampleLogger")
    return logger
    
    # return setup_logger()

def setup_logger():
    logging.config.fileConfig(C.LOG_CONFIG_DIR)
    logger = logging.getLogger("sampleLogger")
    logger.debug("In " + os.uname()[1])
    return logger

def get_stability(in_measure):
    in_measure = np.array(in_measure)
    stability = [np.divide(in_measure[i] - in_measure[i + 1],
                           in_measure[i]) for i in range(in_measure.shape[0] - 1)]
    return stability


def get_run_dir(args):
    # control = "no_cntr/" if args.control_on == 0 else "cntr" + "/" +\
    #             ("").join([str(layer) for layer in args.control_layer]) + "/"
    cur_folder = (C.cur_time + "/" if C.cur_time != "" else "")
    run_dir = C.MODEL_ROOT_DIR + args.prune_method + "/" + args.net_arch + "/" +\
              args.net_dataset + "/" + cur_folder
    checkdir(run_dir)
    return run_dir

def get_args():
    parser = argparse.ArgumentParser()
    # parser = ArgumentParser(default_config_files=["config.yaml"],
    #                         env_prefix="APP",
    #                         default_env=True
    #                         )
    # net = parser.add_argument_group("net")
    parser.add_argument('--net_arch',
                        choices=['vgg11', 'vgg16', 'resnet18', 'alexnet',
                                 'densenet', 'googlenet'],
                        default='resnet18',
                        help='Architectures')

    parser.add_argument('--net_pretrained', type=int, default=1,
                        help='Start with a pretrained network?')

    parser.add_argument('--net_dataset', type=str,
                        choices=['CIFAR10', 'MNIST', 'IMAGENET', 'FashionMNIST', 'CIFAR100'],
                        default='CIFAR10', help='Name of dataset')

    # Training options.
    parser.add_argument('--net_train_epochs', type=int, default=100,
                      help='Number of epochs to train for')

    parser.add_argument('--net_train_per_epoch', type=int, default=2,
                      help='Number of epochs to train for')

    parser.add_argument('--net_warmup', type=int, default=20,
                      help='Number of epochs to perform warmup training')

    parser.add_argument('--net_lr', type=float, default=0.01,
                      help='Learning rate')

    parser.add_argument('--net_batch_size', type=int, default=64,
                      help='Batch size')

    parser.add_argument('--net_weight_decay', type=float, default=0.0005,
                      help='Weight decay')

    # controller
    # control = parser.add_argument_group("control")

    parser.add_argument('--control_on', type=int, default=0)

    parser.add_argument('--control_iteration', type=str, default="1",
                      help='Iteration at which the controller is applied')

    parser.add_argument('--control_epoch', type=int, default=1,
                        # dest="control.epoch",
                      help='Epoch at which the controller is applied')

    parser.add_argument('--control_layer', type=str, default="1 2 3 4 5",
                      help='Network layer at which the controller is applied')

    parser.add_argument('--control_type', type=int, default=1,
                      help='1: correlation, 2: connectivity, 3: prev weights')

    parser.add_argument('--exper_imp_total_iter', type=int, default=5,
                      help='Number of iteration at IMP')

    parser.add_argument('--exper_num_trial', type=int, default=3,
                      help='Number of trials')

    parser.add_argument('--exper_acc_thrd', type=int, default=95,
                      help='Threshold accuracy to stop the training loop')

    parser.add_argument('--exper_gpu_id', type=int, default=0,
                        help='gpu number to use')

    # Pruning options.
    parser.add_argument('--prune_ratio', type=float, default=0.2,
                        help='% of neurons to prune per layer')
    parser.add_argument('--prune_p', type=float, default=0.5)
    parser.add_argument('--prune_q', type=float, default=1.0)
    parser.add_argument('--prune_epsilon', type=float, default=1.0)
    parser.add_argument('--prune_type', type=str, default="percentile",
                        help='% of neurons to prune per layer')
    parser.add_argument('--prune_method', type=str, default="lth",
                        choices=["lth", "sap", "giap", "ciap"],
                        help='What type of LTH experiment you are running?')

    # parser.add_argument('--yaml_config', type=str, default="config.ini",
    #                     help="Address to the config file")
    args = parser.parse_args()

    args.control_layer = [int(layer) for layer in args.control_layer.split(" ")]
    args.control_iteration = [int(iteration) for iteration in
                              args.control_iteration.split(" ")]

    run_dir = get_run_dir(args)
    logger.debug(f"In dir: {run_dir}")
    print(f"In dir: {run_dir}")
    logger.debug(yaml.dump(args.__dict__, default_flow_style=False))
    print(yaml.dump(args.__dict__, default_flow_style=False))
    json.dump(args.__dict__, open(run_dir + "exper.json", 'w'), indent=2)
    # yaml.dump(args, stream=open(run_dir + "exper.json", 'w'),
    #           default_flow_style=False, sort_keys=False)
    return args


def get_yaml_args(args):
    if args.yaml_config == "no":
        return
    config_obj = configparser.ConfigParser()
    config_obj.read(args.yaml_config)
    network_conf = config_obj["network"]
    control_conf = config_obj["control"]
    exper_conf = config_obj["experiment"]
    args.arch = network_conf["arch"]
    args.dataset = network_conf["dataset"]
    args.pretrained = int(network_conf["pretrained"])
    args.lr = float(network_conf["lr"])
    args.train_epochs = int(network_conf["train_epochs"])
    args.train_per_epoch = int(network_conf["train_per_epoch"])
    args.warmup_train = int(network_conf["warmup_train"])
    args.batch_size = int(network_conf["batch_size"])
    args.weight_decay = float(network_conf["weight_decay"])
    args.controller = int(control_conf["controller"])
    args.control_type = int(control_conf["control_type"])
    args.control_at_iter = control_conf["control_at_iter"]
    args.control_at_epoch = int(control_conf["control_at_epoch"])
    args.control_at_layer = control_conf["control_at_layer"]
    args.prune_ratio = float(exper_conf["prune_ratio"])
    args.acc_thrd = int(exper_conf["acc_thrd"])
    args.imp_total_iter = int(exper_conf["imp_total_iter"])
    args.experiment_type = exper_conf["type"]
    args.gpu_id = int(exper_conf["gpu_id"])
    args.num_trial = int(exper_conf["num_trial"])

    args.control_at_layer = [int(l) for l in args.control_at_layer.split(" ")]
    args.control_at_iter = [int(l) for l in args.control_at_iter.split(" ")]

    run_dir = get_run_dir(args)
    json.dump(args.__dict__, open(run_dir + "exper.json", 'w'), indent=2)
    logger.debug(f"In dir: {run_dir}")
    logger.debug(yaml.dump(args.__dict__, default_flow_style=False))
    return args

    
