# File that stores all the ffcv beton dataset files required for this project
# This only needs to be run once to generate the files


## model design: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras.md


from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch    

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
from torchvision.transforms import v2
import torchvision
import torch.nn as nn
import torch.nn.functional as F


from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

train_dataset="./data/cifar_train.beton"
val_dataset="./data/cifar_test.beton"

datasets = {
    'train': torchvision.datasets.CIFAR100('./data', train=True, download=True),
    'test': torchvision.datasets.CIFAR100('./data', train=False, download=True)
    }

for (name, ds) in datasets.items():
    path = train_dataset if name == 'train' else val_dataset
    writer = DatasetWriter(path, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)
        
        
        

# helper function load the coarse labels (maps each label to its superclass)
# source: https://github.com/ryanchankh/cifar100coarse/tree/master


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

superclass = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
              ['bottle', 'bowl', 'can', 'cup', 'plate'],
              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
              ['bed', 'chair', 'couch', 'table', 'wardrobe'],
              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
              ['bridge', 'castle', 'house', 'road', 'skyscraper'],
              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
              ['crab', 'lobster', 'snail', 'spider', 'worm'],
              ['baby', 'boy', 'girl', 'man', 'woman'],
              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

datasets["train"].coarse_targets = sparse2coarse(datasets["train"].targets)
datasets["test"].coarse_targets = sparse2coarse(datasets["test"].targets)

# go over all tuples of coarse labels

# Function to remap labels
def remap_labels(labels, mapping):
    return np.array([mapping[label] for label in labels])



################## ONLY DEBUGGING CODE BELOW #####################
# test how much storage it takes to store the entire dataset with this method
# shouldnt differ much with the original dataset on top but does a little, but also only takes 290MB
# test with writing entire ds 
for dataset_name in ["train", "test"]:
    idx = (datasets[dataset_name].coarse_targets >= 0) | (datasets[dataset_name].coarse_targets >= 0)
    # fill code here
    subset_data = [(datasets[dataset_name].data[k], datasets[dataset_name].targets[k]) for k in np.where(idx)[0]]
    # Create a label mapping for the current pair of superclasses
    unique_labels = np.sort(np.unique([label for _, label in subset_data]))
    label_mapping = {label: new_label for new_label, label in enumerate(unique_labels)}
    # Remap labels

    remapped_data = [(image, label_mapping[label]) for image, label in subset_data]

    # Define path for the new subset beton file
    subset_path = f'./data/subsets/{dataset_name}_superclass_all.beton'

    # Write the subset dataset to a new beton file
    writer = DatasetWriter(subset_path, {'image': RGBImageField(), 'label': IntField()})
    writer.from_indexed_dataset(remapped_data)


########### END OF TESTING CODE

for i in range(0,20):
    for j in range(i+1,20):
        for dataset_name in ["train", "test"]:
            idx = (datasets[dataset_name].coarse_targets == i) | (datasets[dataset_name].coarse_targets == j)
            # fill code here
            subset_data = [(datasets[dataset_name].data[k], datasets[dataset_name].targets[k]) for k in np.where(idx)[0]]
            # Create a label mapping for the current pair of superclasses
            unique_labels = np.sort(np.unique([label for _, label in subset_data]))
            label_mapping = {label: new_label for new_label, label in enumerate(unique_labels)}
            # Remap labels
            
            remapped_data = [(image, label_mapping[label]) for image, label in subset_data]

            # Define path for the new subset beton file
            subset_path = f'./data/subsets/{dataset_name}_superclass_{i}_{j}.beton'

            # Write the subset dataset to a new beton file
            writer = DatasetWriter(subset_path, {'image': RGBImageField(), 'label': IntField()})
            writer.from_indexed_dataset(remapped_data)

