import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap
from matplotlib import rc
import numpy as np
import pickle
import torch
import json
import sys
import pprint
import constants as C
np.set_printoptions(precision=2)
# from tueplots import figsizes, fonts
# rc('font',**{'family':'serif','serif':['Times']})
# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
# plt.rcParams.update(fonts.jmlr2001_tex(family="serif"))
plt.rcParams.update({
    "font.family": "serif",
    # "font.sans-serif": "Liberation Sans",
    "font.size": 14.0,
    "font.weight": "bold",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    # "legend.loc": "upper right",
    "axes.labelweight": "bold",
    # "text.usetex": True,
    "savefig.dpi": 100,     # higher resolution output.
    # "pgf.rcfonts": True,
    # "pgf.texsystem": 'pdflatex', # default is xetex
    # "pgf.preamble": [
    #      r"\usepackage[T1]{fontenc}",
    #      # r"\usepackage{unicode-math}",   # unicode math setup
    #      r"\usepackage{mathpazo}"
    #      ]
})

# matplotlib.use('tkagg')

filled_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p',
                    '*', 'h', 'H', 'D', 'd', 'P', 'X']
linestyles = ['-', '--', '-.', ':']

def plot_all_accuracy(accuracies, filename):
    accuracy_list = accuracies.flatten()
    fig, axs = plt.subplots(1, figsize=(8,4))
    xdata = np.arange(0, 2 * len(accuracy_list) + 0, 2)
    axs.set_title("Accuracy of network in IMP")
    axs.set(xlabel="Training epochs", ylabel="Accuracy(\%)")

    axs.plot(xdata, accuracy_list, 'k')
    fig.tight_layout(pad=2.0)
    # axs.set_xticks(xdata)#, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(0, 2 * len(accuracy_list) + 0, 4)
    axs.set_xticks(major_ticks)
    axs.set_xlim([0, 2 * len(accuracy_list)])
    plt.grid()
    plt.savefig(filename + ".png")


def plot_multi_all_accuracy(accuracies, labels, args, filename):
    # accuracies_list = []
    # for i in range(len(accuracies)):
    #     accuracies_list.append(accuracies[i].flatten())
    fig, axs = plt.subplots(1, figsize=(8,4))
    xdata = np.arange(0, args['net_train_per_epoch'] * args['net_train_epochs'] *
                      len(accuracies[0]), args['net_train_per_epoch'])
    axs.set_title("Accuracy of network in IMP")
    axs.set(xlabel="Training epochs", ylabel="Accuracy(\%)")
    for i in range(len(accuracies)):
        axs.plot(xdata, accuracies[i].flatten(),
                 linestyle=linestyles[i % len(linestyles)],
                 label=labels[i])
                 # alpha=.5) 

    fig.tight_layout(pad=2.0)
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(0, args['net_train_per_epoch']* args['net_train_epochs'] *
                            len(accuracies[0]),
                            args['net_train_per_epoch'] * args['net_train_epochs'])
    axs.set_xticks(major_ticks)
    axs.set_xlim([0, args['net_train_per_epoch']* args['net_train_epochs'] * len(accuracies[0])])
    plt.legend()
    plt.grid()
    plt.savefig(filename + ".png")


def plot_experiment(train_acc, ydata, filename):
    fig, axs = plt.subplots(2)
    discard = 0
    xdata = np.arange(len(train_acc))
    axs[0].scatter(xdata, train_acc, marker=(5, 0))
    axs[0].set_title("Accuracy of network in training")
    axs[0].set(xlabel="IMP Iteration", ylabel="Training accuracy")
    axs[0].set_xticks(xdata)
    # axs[0].legend(loc="upper right")
    xdata = np.arange(discard, len(ydata[0]))
    for i in range(len(ydata)):
        axs[1].plot(xdata, ydata[i][discard:],
                       # marker=(5, i),
                       marker=filled_markers[i],
                       label="Itr. " + str(i),
                       alpha=.5)
        # axs[1].set_yscale('log')
    # axs.set_xlim([0, 1])
    axs[1].set_ylim([-5, 5])
    axs[1].set_title("Correlations between layers")
    axs[1].set(xlabel="Layers", ylabel="Correlation")
    # axs[1].set_xticks(xdata, labels=[str(i) + "-" + str(i + 1) for i in range(len(xdata))])
    axs[1].legend(loc="lower right")
    fig.tight_layout(pad=2.0)
    plt.savefig(filename + ".png")
    # plt.show()


def plot_max_accuracy(accuracies, labels, filename):
    fig, axs = plt.subplots(1, figsize=(5,5))
    xdata = np.arange(1, len(accuracies[0]) + 1)
    # axs.set_title("Accuracy of network in IMP")
    axs.set(xlabel="Iteration", ylabel="Accuracy(\%)")

    for i in range(len(accuracies)):
        axs.plot(xdata, accuracies[i], marker=filled_markers[i],
                 linestyle=linestyles[i % len(linestyles)],
                 label=labels[i],
                 alpha=.5) 
    # axs.plot(xdata, accuracy_list, 'k')
    fig.tight_layout(pad=2.0)
    plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(1, len(accuracies[0]) + 1)
    axs.set_xticks(major_ticks)
    # axs.set_xlim([1, len(accuracies[0])])
    # plt.grid()
    plt.savefig(filename + ".png")


def plot_three():
    """Plot the average of three experiments."""
    exper_folder = sys.argv[1:4]
    acc_list = []
    corr_list = []
    for i in range(3):
        acc, corr = get_vars(C.OUTPUT_DIR + exper_folder[i])
        acc_list.append(acc)
        corr_list.append(corr)

    acc_mean = np.mean(np.array(acc_list), axis=0)
    acc_max = np.max(acc_mean, axis=1)
    corr_mean = np.mean(np.array(corr_list), axis=0)
    plot_all_accuracy(acc_mean, C.OUTPUT_DIR + exper_folder[0] + "three_accuracies")
    plot_experiment(acc_max, corr_mean, C.OUTPUT_DIR + exper_folder[0] + "three_vgg16_correlation")


def plot_train_epochs(epochs, labels, filename):
    fig, axs = plt.subplots(1, figsize=(5,5))
    xdata = np.arange(1, len(epochs[0]) + 1)
    # axs.set_title("Epochs needed to reach to baseline accuracy")
    axs.set(xlabel="Iteration", ylabel="Epochs")

    for i in range(len(epochs)):
        axs.plot(xdata, epochs[i], marker=filled_markers[i],
                 linestyle=linestyles[i % len(linestyles)],
                 label=labels[i],
                 alpha=.5) 
    # axs.plot(xdata, accuracy_list, 'k')
    fig.tight_layout(pad=2.0)
    plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(1, len(epochs[0]) + 1)
    axs.set_xticks(major_ticks)
    # axs.set_xlim([1, len(epochs[0])])
    # plt.grid()
    plt.savefig(filename + ".png")


def plot_connectivity(conns, filename):
    fig, axs = plt.subplots(1, figsize=(12, 8))
    xdata = np.arange(1, len(conns[0]) + 1)

    for i in range(len(conns)):
        axs.plot(xdata, conns[i], marker=filled_markers[i],
                 linestyle=linestyles[i % len(linestyles)],
                 label=f"Iter {i}",
                 alpha=.5) 
    fig.tight_layout(pad=2.0)
    plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(1, len(conns[0]) + 1)
    axs.set_xticks(major_ticks)
    # axs.set_xlim([1, len(epochs[0])])
    axs.set_title("Accuracy of network in training")
    axs.set(xlabel="Layer index", ylabel="Correlations")
    plt.grid()
    plt.savefig(filename + ".png")


def plot_correlations(filename):
    corrs = pickle.load(open(filename, "rb"))
    c_colors = plt.get_cmap('YlGnBu')
    values = np.linspace(0, 1, 31)
    colors = c_colors(values)

    fig, axs = plt.subplots(5, figsize=(10, 18))
    xdata = np.arange(1, len(corrs[0][0]) + 1)
    major_ticks = np.arange(1, len(corrs[0][0]) + 1)

    for i in range(len(corrs[0])):
        axs[(i+1)//31].plot(xdata, [torch.norm(corrs[0][i][layer]) for layer in
                            range(len(corrs[0][i]))],
                            # label=f"Iter {(i+1)//31}",
                            c=colors[i % 31],
                            lw=1,
                            alpha=.9)

    for i in range(5):
        axs[i].set_xticks(major_ticks)
        axs[i].set_title(f"Iter {i}")
        axs[i].set_ylim(bottom=0, top=500)
        axs[i].set_xlim(left=1, right=len(corrs[0][0]))
        axs[i].set(xlabel="Layer index", ylabel="Norm Correlations")
        axs[i].grid()
    fig.tight_layout(pad=2.0)
    # plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    # plt.grid()
    plt.savefig(filename[:-4] + ".png")
    # c_greens = plt.get_cmap('Greens')
    # c_reds = plt.get_cmap('Reds')
    # c_purples = plt.get_cmap('Purples')
    # c_greys = plt.get_cmap('Greys')
    # blues, greens, reds, purples, greys = c_blues(values), c_greens(values),
    # c_reds(values), c_purples(values), c_greys(values)


def plot_ablation(exper_dirs):
    # GIAP
    # labels = ["0.3-(1, 2)", "0.4-(1, 2)", "0.5-(1, 2)", "0.6-(1, 2)", "0.7-(1, 2)",
    #           "0.3-(0.5, 1)", "0.4-(0.5, 1)", "0.5-(0.5, 1)", "0.6-(0.5, 1)",
    #           "0.7-(0.5, 1)"]

    labels = ["0.08-(1, 2)", "0.10-(1, 2)", "0.11-(1, 2)", "0.12-(1, 2)", "0.15-(1, 2)",
              "0.08-(0.5, 1)", "0.10-(0.5, 1)", "0.11-(0.5, 1)", "0.12-(0.5, 1)",
              "0.15-(0.5, 1)"]

    acc_dict = {}
    last_acc_dict = {}
    # last_acc_error_dict = {}
    comp_dict = {}
    last_inds_dict = {}
    # last_inds_error_dict = {}

    c_colors = plt.get_cmap("turbo")
    values = np.linspace(0, 1, len(exper_dirs))
    # values = np.linspace(0, 1, len(exper_dirs) + 6)
    # remove_i = np.arange(len(values)//2 - 3, len(values) // 2 + 3)
    # values = np.delete(values, remove_i)
    colors = c_colors(values)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # read the accuracies array for all experiments
    for exp_ind, exp_dir in enumerate(exper_dirs):
        acc_dict[labels[exp_ind]] = pickle.load(open(exp_dir + "accuracies.pkl", "rb"))
        comp_dict[labels[exp_ind]] = np.mean(pickle.load(open(exp_dir +
                                                              "comp_levels.pkl",
                                                              "rb")), axis=0)

    # process the ciap and giap experiments
    for acc in acc_dict:
        # accuracy of giap and ciap is the last non-zero element
        last_inds_dict[acc] = np.zeros([len(acc_dict[acc]), acc_dict[acc][0].shape[0]])
        last_acc_dict[acc] = np.zeros([len(acc_dict[acc]), acc_dict[acc][0].shape[0]])

        for trial in range(len(acc_dict[acc])):
            for iteration in range(acc_dict[acc][trial].shape[0]):
                for epoch in range(acc_dict[acc][trial][iteration].shape[0]):
                    if abs(acc_dict[acc][trial][iteration][epoch]) < .001:
                        break
                    last_inds_dict[acc][trial][iteration] = epoch+1
                    last_acc_dict[acc][trial][iteration] = acc_dict[acc][trial][iteration][epoch]

        # last_acc_error_dict[acc] = np.std(last_acc_dict[acc], axis=0)
        # last_inds_error_dict[acc] = np.std(last_inds_dict[acc], axis=0)

        last_acc_dict[acc] = np.mean(last_acc_dict[acc], axis=0)
        last_inds_dict[acc] = np.mean(last_inds_dict[acc], axis=0)

    x_acc = np.arange(1, len(comp_dict[labels[0]]) + 1)
    print("accuracy")
    pprint.pprint(last_acc_dict)
    print("last index")
    pprint.pprint(last_inds_dict)
    print("remaining weights")
    pprint.pprint(comp_dict)
    # pprint.pprint(last_acc_error_dict)
    # pprint.pprint(last_inds_error_dict)

    # plot the performance vs. iteration
    for ind, acc in enumerate(last_acc_dict):
        axs[0].plot(x_acc, last_acc_dict[acc], label=acc, c=colors[ind],
                    marker='o')

    axs[0].set(xlabel="Iteration", ylabel="Accuracy")
    axs[0].set_title("Performance Comparison")
    axs[0].legend(title="epsilon-(p, q)")
    axs[0].set_ylim(bottom=89., top=95.)

    # plot number of epochs vs. iteration
    for ind, inds in enumerate(last_inds_dict):
        axs[1].plot(x_acc, last_inds_dict[inds], c=colors[ind], marker='o')

    axs[1].set(xlabel="Iteration", ylabel="Epochs")
    axs[1].set_title("# Training epochs in each iterations")
    axs[1].legend()

    # plot the remaining weight vs. iteration
    # for ind, comp in enumerate(comp_dict):
    axs[2].plot(x_acc, comp_dict[labels[0]], label="(p, q)=(1, 2)", c="purple", marker='o')
    axs[2].plot(x_acc, comp_dict[labels[6]], label="(p, q)=(0.5, 1)", c="green", marker='o')
    # axs[2].plot(x_acc, comp_dict["LTH"], label="LTH", c="cyan", marker='o')

    axs[2].set(xlabel="Iteration", ylabel="Remaining Weights %")
    axs[2].set_title("Remaining weights")
    axs[2].legend()

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    fig.tight_layout(pad=2.0)
    out_dir = "../output/figures/ablation-" + "-".join(exper_dirs[0].split("/")[3:5])
    print("saved in:", out_dir)
    plt.savefig(out_dir + ".png")

    # axs[0, 1].set_xticks(major_ticks)
    # axs[0, 1].set_title(f"Iter {i}")
    # axs[0, 1].set_ylim(bottom=-0.05, top=.4)
    # axs[0, 1].set_xlim(left=1, right=len(similarity[0][0]))
    # axs[0, 1].set(xlabel="Layer index", ylabel="Connectivity")


def plot_accuracy(exper_dirs):
    labels = ["LTH", "SAP(1, 2)", "SAP(0.5, 1)", "InCoP-IF(1, 2)", "InCoP-IF(0.5, 1)",
              "InCoP-GF(1, 2)", "InCoP-GF(0.5, 1)"]
    acc_dict = {}
    last_acc_dict = {}
    last_acc_error_dict = {}
    comp_dict = {}
    last_inds_dict = {}
    last_inds_error_dict = {}

    c_colors = plt.get_cmap("Set1")
    values = np.linspace(0, 1, len(exper_dirs))
    # values = np.linspace(0, 1, len(exper_dirs) + 6)
    # remove_i = np.arange(len(values)//2 - 3, len(values) // 2 + 3)
    # values = np.delete(values, remove_i)
    colors = c_colors(values)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # read the accuracies array for all experiments
    for exp_ind, exp_dir in enumerate(exper_dirs):
        acc_dict[labels[exp_ind]] = pickle.load(open(exp_dir + "accuracies.pkl", "rb"))
        comp_dict[labels[exp_ind]] = np.mean(pickle.load(open(exp_dir +
                                                              "comp_levels.pkl",
                                                              "rb")), axis=0)

    # process the ciap and giap experiments
    for acc in acc_dict:
        if "incop" in acc.lower():
            # accuracy of giap and ciap is the last non-zero element
            last_inds_dict[acc] = np.zeros([len(acc_dict[acc]), acc_dict[acc][0].shape[0]])
            last_acc_dict[acc] = np.zeros([len(acc_dict[acc]), acc_dict[acc][0].shape[0]])

            for trial in range(len(acc_dict[acc])):
                for iteration in range(acc_dict[acc][trial].shape[0]):
                    for epoch in range(acc_dict[acc][trial][iteration].shape[0]):
                        if abs(acc_dict[acc][trial][iteration][epoch]) < .001:
                            break
                        last_inds_dict[acc][trial][iteration] = epoch+1
                        last_acc_dict[acc][trial][iteration] = acc_dict[acc][trial][iteration][epoch]

            last_acc_error_dict[acc] = np.std(last_acc_dict[acc], axis=0)
            last_inds_error_dict[acc] = np.std(last_inds_dict[acc], axis=0)

            last_acc_dict[acc] = np.mean(last_acc_dict[acc], axis=0)
            last_inds_dict[acc] = np.mean(last_inds_dict[acc], axis=0)

        else:
            # accuracy of SAP and lth is the last element
            acc_dict[acc] = np.mean(acc_dict[acc], axis=0)
            sap_len = acc_dict[acc].shape[1]
            last_acc_dict[acc] = np.array([accur[-1] for accur in acc_dict[acc]])
            last_inds_dict[acc] = [sap_len] * acc_dict[acc].shape[0]
            # comp_dict[acc] = np.mean(pickle.load(open(exp_dir + "comp_levels.pkl", "rb")), axis=0)
            last_acc_error_dict[acc] = np.array([0. for accur in acc_dict[acc]])
            last_inds_error_dict[acc] = [0] * acc_dict[acc].shape[0]

    x_acc = np.arange(1, len(comp_dict[labels[0]]) + 1)
    # pprint.pprint(last_acc_dict)
    # pprint.pprint(last_inds_dict)
    # pprint.pprint(comp_dict)
    # pprint.pprint(last_acc_error_dict)
    # pprint.pprint(last_inds_error_dict)

    # plot the performance vs. iteration
    for ind, acc in enumerate(last_acc_dict):
        axs[0].errorbar(x_acc, last_acc_dict[acc],
                        yerr=last_acc_error_dict[acc],
                        label=acc, c=colors[ind], marker='o')

    axs[0].set(xlabel="Iteration", ylabel="Accuracy")
    axs[0].set_title("Performance Comparison")
    axs[0].legend()
    # axs[0].set_ylim(bottom=98., top=100.)

    # plot number of epochs vs. iteration
    for ind, inds in enumerate(last_inds_dict):
        axs[1].errorbar(x_acc, last_inds_dict[inds],
                        yerr=last_inds_error_dict[inds], label=inds,
                        c=colors[ind], marker='o')

    axs[1].set(xlabel="Iteration", ylabel="Epochs")
    axs[1].set_title("# Training epochs in each iterations")
    axs[1].legend()

    # plot the remaining weight vs. iteration
    # for ind, comp in enumerate(comp_dict):
    axs[2].plot(x_acc, comp_dict[labels[0]], label="LTH", c="cyan", marker='o')
    axs[2].plot(x_acc, comp_dict[labels[3]], label="(p, q)=(1, 2)", c="purple", marker='o')
    axs[2].plot(x_acc, comp_dict[labels[4]], label="(p, q)=(0.5, 1)", c="green", marker='o')

    axs[2].set(xlabel="Iteration", ylabel="Remaining Weights %")
    axs[2].set_title("Remaining weights")
    axs[2].legend()

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    fig.tight_layout(pad=2.0)
    # fig.suptitle((" with ".join(exper_dirs[0].split("/")[3:5])).title(), fontsize=20)
    out_dir = "../output/figures/" + "-".join(exper_dirs[0].split("/")[3:5])
    print("saved in:", out_dir)
    plt.savefig(out_dir + ".png")

    # axs[0, 1].set_xticks(major_ticks)
    # axs[0, 1].set_title(f"Iter {i}")
    # axs[0, 1].set_ylim(bottom=-0.05, top=.4)
    # axs[0, 1].set_xlim(left=1, right=len(similarity[0][0]))
    # axs[0, 1].set(xlabel="Layer index", ylabel="Connectivity")

def read_variables(exper_dir):
    all_accuracy = pickle.load(open(exper_dir + "accuracies.pkl", "rb"))
    comp_level = pickle.load(open(exper_dir + "comp_levels.pkl", "rb"))
    similarity = pickle.load(open(exper_dir + "similarity.pkl", "rb"))
    corrs = pickle.load(open(exper_dir + "corrs.pkl", "rb"))
    grads = pickle.load(open(exper_dir + "grads.pkl", "rb"))
    return all_accuracy, comp_level, similarity, corrs, grads


def plot_similarity(exper_dir, vars=None):
    args = json.loads(open(exper_dir + "exper.json", "rb").read())
    train_epochs = args["net_train_epochs"] + 1
    imp_num = args["exper_imp_total_iter"]
    if vars is None:
        acc, comp_level, sim, corrs, grads = read_variables(exper_dir)

    exper_len = np.arange(1, len(acc[0][0]) + 1)
    fig, axs = plt.subplots(imp_num, 5, figsize=(16, 9))
                            # gridspec_kw={'width_ratios': [10, 10, 10]})
    network_len = len(sim[0][0])
    net_layers = np.arange(1, network_len + 1)
    c_colors = plt.get_cmap("coolwarm")
    values = np.linspace(0, 1, train_epochs)  # len(sim[0]))
    colors = c_colors(values)
    major_ticks = np.arange(1, network_len + 1)

    # axs[0, 0].axis("off")
    # cmap = ListedColormap(colors)
    # cbar = ColorbarBase(ax=axs[0, 0], cmap=cmap, ticks=np.arange(0, 1.1, .2))
    # cbar.set_ticklabels(np.arange(0, train_epochs, train_epochs // 5))
    rho_opt = torch.Tensor([elem.mean() for elem in corrs[0][train_epochs - 1]])
    # import ipdb; ipdb.set_trace()
    # tmp = [(torch.Tensor([elem.mean() for elem in corrs[0][0 * train_epochs + j]])
    #         - rho_opt).norm().item() for j in range(train_epochs)]

    for i in range(imp_num):
        print([torch.Tensor([elem.mean()
                             for elem in corrs[0][i * train_epochs + j]
                             ]).norm().item() for j in range(train_epochs)])
        axs[i, 0].plot(np.arange(train_epochs), [(torch.Tensor(
            [elem.mean() for elem in corrs[0][i * train_epochs + j]]) - rho_opt
                                 ).norm().item() for j in range(train_epochs)])

    # similarities
    # print("similarity:", len(sim[0]))
    # for i in range(len(sim[0])):
    #     axs[(i // train_epochs) + 1, 0].plot(net_layers, sim[0][i],
    #                                            label=f"Iter {(i+1 % train_epochs)}",
    #                                            c=colors[i % (train_epochs)])

    # axs[0, 0].axis("off")
    for i in range(imp_num):
        # axs[i, 0].set_xticks(major_ticks)
        axs[i, 0].set_title(f"Iter {i}")
        # axs[i + 1, 0].set_ylim(bottom=0.01, top=.02)
        # axs[i + 1, 0].set_xlim(left=1, right=len(sim[0][0]))
        # axs[i + 1, 0].set(xlabel="Layer index", ylabel="Similarity")
        axs[i, 0].grid()

    # connectivity
    print("connectivity:", len(corrs[0]))
    # for i in range(train_epochs + 1):
    #     axs[0, 1].plot(net_layers[:-1],
    #                    [elem.mean() for elem in corrs[0][i]],
    #                    label=f"Iter {(i+1 % train_epochs)}",
    #                    c=colors[i % (train_epochs + 2)])

    for i in range(len(corrs[0])):
        axs[(i // (train_epochs)), 1].plot(net_layers[:-1],
                                               # corrs[0][i],
                                               [elem.mean() for elem in corrs[0][i]],
                                               label=f"Iter {(i+1 % train_epochs)}",
                                               c=colors[i % train_epochs])

    for i in range(imp_num):
        # axs[i, 1].set_xticks(major_ticks)
        axs[i, 1].set_title(f"Iter {i}")
        # axs[i, 1].set_ylim(bottom=-0.05, top=.4)
        # axs[i + 1, 1].set_xlim(left=1, right=len(similarity[0][0]))
        axs[i, 1].set(xlabel="Layer index", ylabel="Connectivity")
        axs[i, 1].grid()

    # Gradient flow
    print("gradient:", len(grads[0]))
    grad_network_len = len(grads[0][0][0])
    net_layers = np.arange(1, grad_network_len + 1)
    for i in range(len(grads[0])):
        axs[(i // (train_epochs)), 2].plot(net_layers,
                                           grads[0][i][0],
                                           # [elem.abs().mean() for elem in grads[0][i].values()],
                                           label=f"Iter {(i+1 % train_epochs)}",
                                           c=colors[i % train_epochs])

        axs[(i // (train_epochs)), 3].plot(net_layers,
                                           # [elem.abs().norm() for elem in grads[0][i].values()],
                                           grads[0][i][1],
                                           label=f"Iter {(i+1 % train_epochs)}",
                                           c=colors[i % train_epochs])

    for i in range(imp_num):
        # axs[i, 2].set_xticks(major_ticks)
        axs[i, 2].set_title(f"Iter {i}")
        # axs[i, 2].set_ylim(bottom=0.0001, top=.02)
        axs[i, 2].set_xlim(left=1, right=grad_network_len)
        axs[i, 2].set(xlabel="Layer index", ylabel="Gradient Mean")
        axs[i, 2].grid()

        # axs[i, 3].set_xticks(major_ticks)
        axs[i, 3].set_title(f"Iter {i}")
        # axs[i, 3].set_ylim(bottom=0.0001, top=.02)
        axs[i, 3].set_xlim(left=1, right=grad_network_len)
        axs[i, 3].set(xlabel="Layer index", ylabel="Gradient Norm")
        axs[i, 3].grid()

    # Accuracy
    print("accuracy: ", len(acc[0][0]))
    for i in range(imp_num):
        axs[i, 4].plot(exper_len, acc[0][i], 'k')
        axs[i, 4].set_title(f"Rem. Weights {comp_level[0][i]}")
        axs[i, 4].set_ylim(bottom=70, top=100)
        axs[i, 4].set(xlabel="Training Epoch", ylabel="Accuracy")
        axs[i, 4].grid()
        # axs[i, 1].set_xticks(major_ticks)
        # axs[i, 1].set_xlim(left=1, right=len(similarity[0][0]))

    # plt.legend()
    # cbar.set_label()

    # for i in range(1, 4):
    #     axs[i, 2].axis("off")

    fig.tight_layout(pad=2.0)

    # axs.set_title("Accuracy of network in training")
    plt.savefig(exper_dir + "similarity.png")


def main():
    # plot_accuracy()
    # plot_correlations(sys.argv[1])
    # plot_similarity(sys.argv[1])
    plot_accuracy(sys.argv[1:])
    # plot_ablation(sys.argv[1:])


if __name__ == '__main__':
    main()
