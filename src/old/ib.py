from utils import load_mnist
import utils
import numpy as np
from mlp import Layer, LayerArgs, Model, ModelArgs
from collections import Counter
import math
from mi_tool import MI
import plot_tool

X_train, y_train = load_mnist('/home/soheil/Documents/fashion-mnist/data/fashion/', kind='train')
X_test, y_test = load_mnist('/home/soheil/Documents/fashion-mnist/data/fashion', kind='t10k')

# normalize inputs
X_train, X_test = np.multiply(X_train, 1.0 / 255.0), np.multiply(X_test, 1.0 / 255.0)
X_train, y_train = utils.unison_shuffled_copies(X_train, y_train)
X_train_subset, y_train_subset = X_train[:10000], y_train[:10000]

layer_args = [LayerArgs(784, 784, layer_type = "INPUT"), \
              LayerArgs(784, 30), \
              LayerArgs(30, 20), \
              LayerArgs(20, 15), \
              LayerArgs(15, 10, layer_type = "OUTPUT", activate = np.exp)]

model_args = ModelArgs(num_passes = 80, max_iter=100000, report_interval=500)
model = Model(layer_args, model_args)
model.feed_data(X_train, y_train, X_test, y_test)
model.trial_data(X_train_subset, y_train_subset)
model.intialize_model()

MI_client = MI(X_train_subset, y_train_subset, 10)
MI_client.discretize()
MI_client.pre_compute()

for epoch, hidden_layers in model.run_model():
    MI_client.mi_single_epoch(hidden_layers, epoch)


