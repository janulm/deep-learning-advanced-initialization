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
                
                del idx, subset_data, unique_labels, label_mapping, remapped_data, writer
                


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
        
        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline},
                               os_cache=True)

    return loaders, start_time


import matplotlib.pyplot as plt

def plot_training(tracked_params,name,plot=True, save=False):
    # Plot the training curves
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    desc = f"Model was trained for {tracked_params['epochs']} epochs, with a weight decay of {tracked_params['weight_decay']}, a learning rate of {tracked_params['lr']} and a momentum of {tracked_params['momentum']} and reduce factor of {tracked_params['reduce_factor']}."
    # plot the training loss together with the learning rate
    # compute the x-axis for the train loss, sampled every epoch\
    x1 = np.arange(0, len(tracked_params['train_loss']), 1)
    axs[0].plot(x1,tracked_params['train_loss'], label='train_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(desc)
    # add the val_loss to the plot
    # compute the x-axis for the val loss, sampled every "tracking_freq" epoch
    x2 = np.arange(0, len(tracked_params['train_loss']), tracked_params['tracking_freq'])
    axs[0].plot(x2, tracked_params['val_loss'], label='val_loss')
    axs[0].legend(loc=1)
    
    # add the learning rate to the plot
    ax2 = axs[0].twinx()
    ax2.plot(tracked_params['lr_list'], label='learning_rate', color='red')
    ax2.set_ylabel('Learning Rate') 
    # fix that on this plot the legends are overlapping
    
    # plot the training accuracy
    axs[1].plot(tracked_params['train_acc_top1'], label='train_acc')
    # plot the validation accuracy
    axs[1].plot(tracked_params['val_acc_top1'], label='val_acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title(desc)
    axs[1].legend()
    # add some spacing between plots 
    if save: 
        #plt.savefig(f'./plots/{name}.png')
        fig.savefig(f'./plots/{name}.png')
    if plot:    
        plt.show()
    
    # free up memory
    fig.clear()
    plt.close(fig)
    # add the val_loss to the plot
    

def train(model, loaders, lr=0.1, epochs=100, momentum=0.9, weight_decay=0.0001, reduce_patience=5, reduce_factor=0.2, tracking_freq=5,early_stopping_patience=10, early_stopping_min_epochs=100, do_tracking=True, verbose=False):
    # dictionary to keep track of training params and results
    train_dict = {}
    train_dict['lr'] = lr
    train_dict['epochs'] = epochs
    train_dict['momentum'] = momentum
    train_dict['weight_decay'] = weight_decay
    train_dict['reduce_patience'] = reduce_patience
    train_dict['reduce_factor'] = reduce_factor
    train_dict['tracking_freq'] = tracking_freq
    # results
    # training loss is tracked every epoch
    train_dict['train_loss'] = []
    train_dict['val_loss'] = []
    train_dict['lr_list'] = []
    train_dict['train_acc_top1'] = []
    train_dict['train_acc_top5'] = []
    train_dict['val_acc_top1'] = []
    train_dict['val_acc_top5'] = []

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = ch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=reduce_patience, verbose=verbose, factor=reduce_factor)
    len_train_loader = len(loaders['train'])
    len_val_loader = len(loaders['test'])

    best_val_acc = 0
    early_stopping_counter = 0

    for i in tqdm(range(epochs),disable= not verbose):
        model.train()
        running_loss = 0.0
        total_correct, total_num, total_correct_top5 = 0., 0., 0.

        for ims, labs in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = criterion(out, labs)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if do_tracking and (i+1)%tracking_freq == 0: # only do bookkeeping if needed
                # computing top1 accuracy
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
                # computing top5 accuracy
                total_correct_top5 += out.argsort(1)[:,-5:].eq(labs.unsqueeze(-1)).sum().cpu().item()
                
        # save training loss
        if verbose: print(f'Epoch {i+1}/{epochs}, Training Loss: {running_loss/len_train_loader}')
        train_dict['train_loss'].append(running_loss/len_train_loader)
        # keep track of the current lr 
        train_dict['lr_list'].append(optimizer.param_groups[0]['lr'])
        # keep track of other metrics
        if do_tracking and (i+1)%tracking_freq == 0:
            train_top1 = total_correct / total_num * 100
            train_top5 = total_correct_top5 / total_num * 100
            val_loss = 0.0
            total_val_correct, total_val_num, total_val_correct_top5 = 0., 0., 0.
            model.eval()
            with ch.no_grad():
                for val_ims, val_labs in loaders['test']:
                    val_out = model(val_ims)
                    val_loss += criterion(val_out, val_labs).item()
                    # computing top1 accuracy
                    total_val_correct += val_out.argmax(1).eq(val_labs).sum().cpu().item()
                    total_val_num += val_ims.shape[0]
                    # computing top5 accuracy
                    total_val_correct_top5 += val_out.argsort(1)[:,-5:].eq(val_labs.unsqueeze(-1)).sum().cpu().item()
            val_loss /= len_val_loader
            val_top1 = total_val_correct / total_val_num * 100
            scheduler.step(running_loss)
            val_top5 = total_val_correct_top5 / total_val_num * 100
            train_dict['val_loss'].append(val_loss)
            train_dict['train_acc_top1'].append(train_top1)
            train_dict['train_acc_top5'].append(train_top5)
            train_dict['val_acc_top1'].append(val_top1)
            train_dict['val_acc_top5'].append(val_top5)
            if verbose: print(f'Epoch {i+1}/{epochs}, Validation Loss: {val_loss}')
            if i > early_stopping_min_epochs:
                # Early stopping based on increasing validation loss
                if val_top1 < best_val_acc:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        if verbose: print(f"Early stopping triggered at epoch {i}!")
                        return model, train_dict
                else:
                    best_val_acc = val_top1
                    early_stopping_counter = 0

    return model, train_dict