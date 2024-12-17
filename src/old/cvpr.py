from plot_tool import *
from utils import *


def main():
    labels = [r"w. $\varphi$", r"w.o. $\varphi$"]
    latex_dir = "/home/soheil/Sync/umaine/bnn/report/cvpr/figures/"

    # Performance experiment
    arch = "vgg16"
    exper_1_dirs = ["06-03-13-15/", "06-03-14-12/", "07-03-01-37/"]
    exper_1_acc = get_max_accuracy(arch, exper_1_dirs)
    exper_1_mean_acc = get_mean_accuracy("vgg16", exper_1_dirs)

    exper_2_dirs = ["07-03-02-12/", "07-03-09-42/", "07-03-09-44/"]
    exper_2_acc = get_max_accuracy(arch, exper_2_dirs)

    exper_3_dirs = ["07-03-02-13/", "07-03-09-40/", "07-03-09-41/"]
    exper_3_acc = get_max_accuracy(arch, exper_3_dirs)

    exper_4_dirs = ["24-02-17-55/", "27-02-17-09/", "27-02-18-30/"]
    exper_4_acc = get_max_accuracy(arch, exper_4_dirs)
    exper_4_mean_acc = get_mean_accuracy(arch, exper_4_dirs)

    plot_max_accuracy([exper_1_acc, exper_4_acc], labels,
                      latex_dir + arch + "-performance-max-acc") 
 
    plot_multi_all_accuracy([exper_1_mean_acc, exper_4_mean_acc],
                      latex_dir + "vgg-all-accuracies") 

    # Resnet
    arch = "resnet"
    # w.o. controller
    # "14-03-20-03-21/","20-03-07-17-21/",
    perf_wo_dirs = ["14-03-00-32/", "14-03-16-36/", "14-03-21-34-30/"] 
    perf_wo_mean_acc = get_mean_accuracy(arch, perf_wo_dirs)
    perf_wo_max_acc = get_max_accuracy(arch, perf_wo_dirs)

    perf_w_dirs = ["20-03-18-56-41/", "20-03-15-22-25/", "21-03-04-42-35/"] 
    perf_w_mean_acc = get_mean_accuracy(arch, perf_w_dirs)
    perf_w_max_acc = get_max_accuracy(arch, perf_w_dirs)

    plot_multi_all_accuracy([perf_w_mean_acc, perf_wo_mean_acc],
                      latex_dir + arch + "-all-accuracies") 

    plot_max_accuracy([perf_w_max_acc, perf_wo_max_acc], labels,
                      latex_dir + arch + "-performance-max-acc") 

    # EFficiency experiments
    # vgg16
    arch = "vgg16"
    vgg_effc_wo_dirs = ["17-03-16-16-51/", "24-03-10-34-02/", "17-03-04-34-08/"]
    #"16-03-08-06-23/", #"17-03-07-24-36/", "16-03-08-03-12/", "24-03-18-01-23/",  
    effc_wo_epoch = get_mean_train_epochs(arch, vgg_effc_wo_dirs)

    vgg_effc_w_dirs = ["22-03-22-01-15/", "24-03-10-34-10/", "25-03-04-35-24/"]
    #, "23-03-09-30-23/", "22-03-20-44-24/"]  "24-03-21-46-24/", 
    effc_w_epoch = get_mean_train_epochs(arch, vgg_effc_w_dirs)

    plot_train_epochs([effc_w_epoch, effc_wo_epoch], labels,
                      latex_dir + arch + "-train-epochs") 

    # resnet
    arch = "resnet"
    effc_wo_dirs = ["20-03-17-46-01/", "21-03-00-07-00/", "20-03-17-46-11/"]
    effc_wo_epoch = get_mean_train_epochs(arch, effc_wo_dirs)

    effc_w_dirs = ["20-03-04-28-28/", "20-03-04-25-48/", "20-03-15-38-23/"]
    effc_w_epoch = get_mean_train_epochs(arch, effc_w_dirs)

    plot_train_epochs([effc_w_epoch, effc_wo_epoch], labels,
                      latex_dir + arch + "-train-epochs") 

if __name__ == '__main__':
    main()
