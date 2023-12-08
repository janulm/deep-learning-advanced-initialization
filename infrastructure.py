############## INFRASTRUCTURE ##############
# This file contains all the imports and definitions of the code that is reused many times in the project.
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device {device}')

def write_cifar100_to_beton(): 
    train_dataset="./data/cifar_train.beton"
    val_dataset="./data/cifar_test.beton"
    datasets = {
    'train': torchvision.datasets.CIFAR100('./data', train=True, download=True),
    'test': torchvision.datasets.CIFAR100('./data', train=False, download=True)
    }

    for (name, ds) in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(write_mode="raw"),
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

def get_cifar_classes():
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
    return superclass

def write_cifar100_superclass_subsets_to_beton():
    # takes every pair of 20 the superclasses 
    # and creates a dataset with the corresponding 10 classes
    # and stores them to disk
    datasets = {
    'train': torchvision.datasets.CIFAR100('./data', train=True, download=True),
    'test': torchvision.datasets.CIFAR100('./data', train=False, download=True)
    }
    datasets["train"].coarse_targets = sparse2coarse(datasets["train"].targets)
    datasets["test"].coarse_targets = sparse2coarse(datasets["test"].targets)

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
                


def make_dataloaders(train_dataset="./data/cifar_train.beton", val_dataset="./data/cifar_test.beton", batch_size=256, num_workers=12,device="cuda"):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }
    start_time = time.time()
    # computed these values earlier and hardcoded them here	
    CIFAR_MEAN = [129.310, 124.108, 112.404]
    CIFAR_STD = [68.2125, 65.4075, 70.4055]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device(device)), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(), # flips the image horizontally with a probability of 0.5
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))), # shifts the image horizontally and vertically by a random amount
                Cutout(4, tuple(map(int, CIFAR_MEAN))), # sets a random square patch of the image to mean
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device(device), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float32), # TODO check what the impact for float16 is (it was the initial value, and why it crashes with float16)	
        ])
        if name == 'train':
            image_pipeline.extend([
                v2.RandomRotation(15),
            ])
        image_pipeline.extend([
            v2.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time