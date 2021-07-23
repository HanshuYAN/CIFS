import os
import sys
import pathlib

import numpy as np
import torch

from models.LeNet import LeNet5
# Demo models lenet5
path_of_this_module = os.path.dirname(sys.modules[__name__].__file__) # the dir including this file
TRAINED_MODEL_PATH = os.path.join(path_of_this_module, "trained_models")

def get_mnist_lenet5_clntrained():
    filename = "xx_mnist_lenet5_clntrained.pt"
    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model.eval()
    model.name = "MNIST LeNet5 standard training"
    return model

def get_mnist_lenet5_advtrained():
    filename = "xx_mnist_lenet5_advtrained.pt"
    model = LeNet5()
    model.load_state_dict(
        torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    model.eval()
    model.name = "MNIST LeNet 5 PGD training according to Madry et al. 2018"
    return model




