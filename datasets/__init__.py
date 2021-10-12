import os
from random import shuffle
import sys
import pathlib

import numpy as np
import torch
from torch.functional import split
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets




#
#   Image classification
#
# path_of_this_module = os.path.dirname(sys.modules[__name__].__file__) # the dir including this file
# DATA_PATH = os.path.join(path_of_this_module, '.benchmarks')
DATA_PATH = os.path.join(pathlib.Path.home(), 'Data/benchmark')

# MNIST
def get_mnist_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_train"
    return loader


def get_mnist_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "mnist_test"
    return loader

# CIFAR10 & 100
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def get_cifar10_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=shuffle, num_workers=4)
    loader.name = "cifar10_train"
    return loader


def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

import random
random.seed(0)

def get_indices_each_class(dataset,num=100):
    indices =  []
    labels = dataset.targets.copy()
    random.shuffle(labels)
    
    for c in range(10):
        n = 0
        for i in range(len(labels)):
            if labels[i] == c:
                indices.append(i)
                n += 1
                if n >= num:
                    break
    return indices

def get_cifar10_test_loader(batch_size, shuffle=False, sample_class=None, num_each_class=None):
    dataset = datasets.CIFAR10(DATA_PATH, train=False, download=True, transform=transform_test)
    if sample_class is not None:
        idx = get_indices(dataset, sample_class)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
    elif num_each_class is not None:
        idx = get_indices_each_class(dataset, num_each_class)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
    else:
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4)
    loader.name = "cifar10_test"
    return loader

def add_noise_tensor(x, param=['G',15]):
    """ param should be [type, value] """
    if param is not '.':
        noise_type = param[0]
        noise_value = int(param[1])

        if noise_type == 'G':
            noises = np.random.normal(loc=0, scale=noise_value, size=x.shape)
        elif noise_type == 'S':
            assert False, 'Please use Guassian Noises.'

        x_noise = x + torch.from_numpy(noises / 255).to(torch.float32)
        x_noise = torch.clamp(x_noise, 0, 1)
        return x_noise
    else:
        return x
        

def get_CIFAR100_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DATA_PATH, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=shuffle, num_workers=4)
    loader.name = "CIFAR100_train"
    return loader

def get_CIFAR100_test_loader(batch_size, shuffle=False, sample_class=None):
    dataset = datasets.CIFAR100(DATA_PATH, train=False, download=True, transform=transform_test)
    if sample_class is not None:
        idx = get_indices(dataset, sample_class)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
    else:
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4)
    loader.name = "CIFAR100_test"
    return loader
    

# SVHN
def get_SVHN_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.SVHN(DATA_PATH, split='train', download=True, transform=transform_train),
        batch_size=batch_size, shuffle=shuffle, num_workers=4)
    loader.name = "SVHN_train"
    return loader


# def get_SVHN_test_loader(batch_size, shuffle=False):
#     loader = torch.utils.data.DataLoader(
#         datasets.SVHN(DATA_PATH, split='test', download=True, transform=transform_test),
#         batch_size=batch_size, shuffle=shuffle, num_workers=4)
#     loader.name = "SVHN_test"
#     return loader

def get_indices_SVHN(dataset,class_name):
    indices =  []
    for i in range(len(dataset.labels)):
        if dataset.labels[i] == class_name:
            indices.append(i)
    return indices

def get_indices_each_class_SVHN(dataset,num=100):
    indices =  []

    for c in range(10):
        n = 0
        for i in range(len(dataset.labels)):
            if dataset.labels[i] == c:
                indices.append(i)
                n += 1
                if n >= num:
                    break
    return indices

def get_SVHN_test_loader(batch_size, shuffle=False, sample_class=None, num_each_class=None):
    dataset = datasets.SVHN(DATA_PATH, split='test', download=True, transform=transform_test)
    if sample_class is not None:
        idx = get_indices_SVHN(dataset, sample_class)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
    elif num_each_class is not None:
        idx = get_indices_each_class_SVHN(dataset, num_each_class)
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
    else:
        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=4)
    loader.name = "SVHN_test"
    return loader


# Fashion-MNIST
def get_FashionMNIST_train_loader(batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "FashionMNIST_train"
    return loader


def get_FashionMNIST_test_loader(batch_size, shuffle=False, sample_class=None):
    dataset = datasets.FashionMNIST(DATA_PATH, train=False, download=True, transform=transforms.ToTensor())
    if sample_class is not None:
        idx = get_indices(dataset, sample_class)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        
    loader.name = "FashionMNIST_test"
    return loader

















def get_madry_et_al_cifar10_train_transform():
    return transforms.Compose([
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


class ImageNetClassNameLookup(object):

    def _load_list(self):
        import json
        with open(self.json_path) as f:
            class_idx = json.load(f)
        self.label2classname = [
            class_idx[str(k)][1] for k in range(len(class_idx))]

    def __init__(self):
        self.json_url = ("https://s3.amazonaws.com/deep-learning-models/"
                         "image-models/imagenet_class_index.json")
        self.json_path = os.path.join(DATA_PATH, "imagenet_class_index.json")
        if os.path.exists(self.json_path):
            self._load_list()
        else:
            import urllib
            urllib.request.urlretrieve(self.json_url, self.json_path)
            self._load_list()


    def __call__(self, label):
        return self.label2classname[label]


def get_panda_image():
    img_path = os.path.join(DATA_PATH, "panda.jpg")
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_panda_image():
        from skimage.io import imread
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_panda_image()
    else:
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_panda_image()
    
