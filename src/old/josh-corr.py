
import os
import numpy as np
import torch
import torch.nn as nn
import copy

import utils
import plot_tool
from data_loader import Data
from network import Network, train, test
import constants as C

class Connectivity:
    def __init__(self, model, dataloader, device):
        "docstring"
        self.visualisation = {}
        self.model = model
        self.data_loader = dataloader
        self.device = device
        self.task_conns = {}
        self.task_conn_aves = {}        
        self.conn_aves = {}
        self.conns = {}
        self.task_num = 0

    def calc_conn(self, parent_key, children_key, key_id):
        self.model.eval()

        # Obtain Activations
        print("----------------------------------")
        print("Collecting activations from layers")

        p1_op = {}
        c1_op = {}
        p1_lab = {}
        c1_lab = {}    

        unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
        act         = {}
        lab         = {}


        ### Get activations and labels from the function in utils prior to
        ### calculating connectivities 
        for item_key in unique_keys:
            act[item_key], lab[item_key] = self.activations(item_key)

        for item_idx in range(len(parent_key)):
            p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
            p1_lab[str(item_idx)] = copy.deepcopy(lab[parent_key[item_idx]]) 
            c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])
            c1_lab[str(item_idx)] = copy.deepcopy(lab[children_key[item_idx]])

        del act, lab


        print("----------------------------------")
        print("Begin Execution of conn estimation")

        parent_aves = []
        p1_op = np.asarray(list(p1_op.values())[0])
        p1_lab = np.asarray(list(p1_lab.values())[0])
        c1_op = np.asarray(list(c1_op.values())[0])
        c1_lab = np.asarray(list(c1_lab.values())[0])

        task = self.task_num

        ### Normalizing the activation values per our proof in the previous work
        for label in list(np.unique(np.asarray(p1_lab))):
            # print("Parent mean and stdev: ", np.mean(p1_op[p1_lab == label]), " ", np.std(p1_op[p1_lab == label]))

            p1_op[p1_lab == label] -= np.mean(p1_op[p1_lab == label])
            p1_op[p1_lab == label] /= np.std(p1_op[p1_lab == label])



            c1_op[c1_lab == label] -= np.mean(c1_op[c1_lab == label])
            c1_op[c1_lab == label] /= np.std(c1_op[c1_lab == label])

        """
        Code for averaging conns by parent prior by layer
        """

        parent_class_aves = []
        parents_by_class = []
        parents_aves = []
        conn_aves = []
        parents = []

        print("p1_op shape: ", p1_op.shape)

        ### Calculating connectivity for each class individually and then
        ### averaging over all classes, as per proofs 
        for c in list(np.unique(np.asarray(p1_lab))):
            p1_class = p1_op[np.where(p1_lab == c)]
            c1_class = c1_op[np.where(c1_lab == c)]

            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            coefs = np.corrcoef(p1_class, c1_class, rowvar=False)

            parents = []
            for i in range(0, len(p1_class[0])):
                parents.append(coefs[i, len(p1_class[0]):])
            parents = np.abs(np.asarray(parents))

            ### This is a growing list of each p-c connectivity for all
            ###     activations of a given class 
            ###     The dimensions are (class, parent, child)
            parents_by_class.append(parents)

        ### Averages all classes, since all class priors are the same for cifar10 and 100
        conn_aves = np.mean(np.asarray(parents_by_class), axis=0)

        ### Then average over the parents and children to get the layer-layer connectivity
        layer_ave = np.mean(conn_aves)

        return layer_ave, conn_aves


    def calc_conns(self):
        self.task_conns = {}
        self.task_conn_aves = {}        

        #!# Probably just calculate the activations here before the looping


        #!# This can still be the same, splitting by parents and children, it just needs to index the acts appropriately rather than generate them each time
        ### Record the indices of all adjacent pairs of layers in the shared network
        ### This numbering method reflects the "layer index" in Figs. 2-4 of the accompanying paper
        parents = []
        children = []
        i = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear):
                if (i == 0):
                    parents.append(module_idx)
                    i += 1
                else:
                    parents.append(module_idx)
                    children.append(module_idx)
        children.append(-1)

        for key_id in range(0, len(parents) - 1): 
            self.task_conn_aves[parents[key_id]], self.task_conns[parents[key_id]] = self.calc_conn([parents[key_id]], [children[key_id]], key_id)

        self.conn_aves[self.task_num] = self.task_conn_aves
        self.conns[self.task_num] = self.task_conns


    ### This was following an implementation by Madan which allowed for parallelization, where we calculate connectivity for one pair of layers at a time
    """
    hook_fn(), activations(), and get_all_layers() adapted from: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
    """

    #### Hook Function
    def hook_fn(self, m, i, o):
        self.visualisation[m] = o 

    def get_act_layer(self, hook_handles, item_key):
    # Create forward hooks to all layers which will collect activation state
        if item_key != -1:
            for module_idx, module in enumerate(self.model.named_modules()):
                if (isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.Linear)):
                    if(module_idx == item_key):
                        hook_handles.append(module[1].register_forward_hook(self.hook_fn))
        else:
            print("classifier hooks")
            # hook_handles.append(self.model.classifier.register_forward_hook(self.hook_fn))


    ### Process and record all of the activations for the given pair of layers
    def activations(self, item_key):
        temp_op       = None
        temp_label_op = None

        parents_op  = None
        labels_op   = None

        handles     = []

        self.get_act_layer(handles, item_key)
        print('Collecting Activations for Layer %s'%(item_key))

        with torch.no_grad():
            for step, data in enumerate(self.data_loader):
                x_input, y_label = data
                self.model(x_input.to(self.device))

                if temp_op is None:
                    temp_op        = self.visualisation[list(self.visualisation.keys())[0]].cpu().numpy()
                    temp_labels_op = y_label.numpy()

                else:
                    temp_op        = np.vstack((self.visualisation[list(self.visualisation.keys())[0]].cpu().numpy(), temp_op))
                    temp_labels_op = np.hstack((y_label.numpy(), temp_labels_op))

                if step % 100 == 0:
                    if parents_op is None:
                        parents_op = copy.deepcopy(temp_op)
                        labels_op  = copy.deepcopy(temp_labels_op)

                        temp_op        = None
                        temp_labels_op = None

                    else:
                        parents_op = np.vstack((temp_op, parents_op))
                        labels_op  = np.hstack((temp_labels_op, labels_op))

                        temp_op        = None
                        temp_labels_op = None


        if parents_op is None:
            parents_op = copy.deepcopy(temp_op)
            labels_op  = copy.deepcopy(temp_labels_op)

            temp_op        = None
            temp_labels_op = None

        else:
            parents_op = np.vstack((temp_op, parents_op))
            labels_op  = np.hstack((temp_labels_op, labels_op))

            temp_op        = None
            temp_labels_op = None

        # Remove all hook handles
        for handle in handles:
            handle.remove()    

        del self.visualisation[list(self.visualisation.keys())[0]]

        ### Average activations for a given filter
        if len(parents_op.shape) > 2:
            parents_op  = np.mean(parents_op, axis=(2,3))

        return parents_op, labels_op


def main():
    # preparing the hardware
    device = utils.get_device()
    args = utils.get_args()
    logger = utils.setup_logger()
    num_exper = 5

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


    corr = []
    test_acc = torch.zeros(num_exper)

    for i in range(num_exper):
        logger.debug("=" * 10 + " experiment " + str(i + 1) + "=" * 10)
        train_acc, _ = train(model, train_dl, loss_fn, optimizer,
                             args.train_epochs, device)
        test_acc[i] = test(model, test_dl, loss_fn, device)

        activations = Connectivity(model, test_dl, device)
        corr.append(activations.calc_conns())
        import ipdb; ipdb.set_trace()

        # utils.save_model(model, C.OUTPUT_DIR, args.arch + f'-{i}-model.pt')
        logger.debug('model is saved...!')

        # utils.save_vars(test_acc=test_acc, corr=corr)

    # plot_tool.plot_connectivity(test_acc, corr)


if __name__ == '__main__':
    main()
