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
from torch.optim import SGD, Adam, lr_scheduler
from torchvision.transforms import v2
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print(f'Using (why does print here device {device}')


#def do_imports_ffcv():
#from fastargs import get_current_config, Param, Section
#from fastargs.decorators import param
#from fastargs.validation import And, OneOf


def write_cifar100_to_beton(): 
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
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
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
                


class CIFAR100Subset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

def get_loaders_cifar100_superclass_subsets_pytorch(superclass1,superclass2,batch_size=256, num_workers=12, normalize=False):
    loaders = {}
    datasets = get_dataset_cifar100_superclass_subsets_pytorch(superclass1,superclass2,normalize)
    for dataset_name in ["train", "test"]:
        dset = datasets[dataset_name]
        should_shuffle = dataset_name == "train"
        if num_workers > 0:
            #loader = DataLoader(dset, batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers, multiprocessing_context="forkserver", persistent_workers=True)
            loader = DataLoader(dset, batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers,persistent_workers=True)
        else:
            loader = DataLoader(dset, batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers)
        loaders[dataset_name] = loader
        # check if this works, not sure if this deletes to much.
        #del dset, loader, should_shuffle
    return loaders


def get_dataset_cifar100_superclass_subsets_pytorch(superclass1,superclass2,normalize=True):
    assert superclass1 != superclass2, "superclass1 and superclass2 must be different"
    assert superclass1 <= 19 and superclass1 >= 0, "superclass1 must be between 0 and 19"
    assert superclass2 <= 19 and superclass2 >= 0, "superclass2 must be between 0 and 19"
    superclass1,superclass2 = min(superclass1,superclass2),max(superclass1,superclass2)

    datasets = {
    'train': torchvision.datasets.CIFAR100('./data', train=True, download=True),
    'test': torchvision.datasets.CIFAR100('./data', train=False, download=True)
    }
    loaders = {}
    datasets["train"].coarse_targets = sparse2coarse(datasets["train"].targets)
    datasets["test"].coarse_targets = sparse2coarse(datasets["test"].targets)
    
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]
    transforms = {}
    if normalize:
        transforms = {
            'train': v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(CIFAR_MEAN, CIFAR_STD)
            ]),
            'test': v2.Compose([ 
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])}
    else:
        transforms = {
            'train': v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                #v2.Normalize(CIFAR_MEAN, CIFAR_STD)
            ]),
            'test': v2.Compose([ 
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                #v2.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])}

    dsets = {}

    for dataset_name in ["train", "test"]:
        # get interger index list where coarse target is superclass1 or superclass2
        idx = (datasets[dataset_name].coarse_targets == superclass1) | (datasets[dataset_name].coarse_targets == superclass2)
        data = datasets[dataset_name].data[idx]
        targets = np.array(datasets[dataset_name].targets)
        targets = targets[idx]

        
        # Create a label mapping for the current pair of superclasses
        unique_labels = np.sort(np.unique(targets))
        label_mapping = {label: new_label for new_label, label in enumerate(unique_labels)}
        # Remap targets: 
        targets_remapped = np.array([label_mapping[label] for label in targets])

        dset = CIFAR100Subset(data, targets_remapped, transform=transforms[dataset_name])
        dsets[dataset_name] = dset
        #del idx, data, targets, unique_labels, label_mapping, targets_remapped
    return dsets


def get_loaders_cifar100_superclass_subsets_ffcv(superclass1,superclass2,batch_size=256, num_workers=12,device="cuda"):
    assert superclass1 != superclass2, "superclass1 and superclass2 must be different"
    assert superclass1 <= 19 and superclass1 >= 0, "superclass1 must be between 0 and 19"
    assert superclass2 <= 19 and superclass2 >= 0, "superclass2 must be between 0 and 19"
    superclass1,superclass2 = min(superclass1,superclass2),max(superclass1,superclass2)

    datasets = {
    'train': torchvision.datasets.CIFAR100('./data', train=True, download=True),
    'test': torchvision.datasets.CIFAR100('./data', train=False, download=True)
    }
    paths = [f'./data/subsets/{dataset_name}_superclass_{superclass1}_{superclass2}.beton' for dataset_name in ["train","test"]]
    loaders, start_time = make_dataloaders_ffcv(paths[0],paths[1],batch_size=batch_size,num_workers=num_workers,device=device)
    return loaders    
    

def make_dataloaders_ffcv(train_dataset="./data/cifar_train.beton", val_dataset="./data/cifar_test.beton", batch_size=256, num_workers=12,device="cuda"):
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
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

def plot_training(tracked_params,name,plot=True, save=False,save_path="./plots/"):
    # Plot the training curves
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    desc = f"Model was trained for {tracked_params['epochs']} epochs, with a weight decay of {tracked_params['weight_decay']}, a learning rate of {tracked_params['lr']} and a momentum of {tracked_params['momentum']} and reduce factor of {tracked_params['reduce_factor']}."
    # plot the training loss together with the learning rate
    # compute the x-axis for the train loss, sampled every epoch\
    # set title for the plot
        
    x1 = np.arange(0, len(tracked_params['train_loss']), 1)
    axs[0].plot(x1,tracked_params['train_loss'], label='train_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(desc)
    # add the val_loss to the plot
    # compute the x-axis for the val loss, sampled every "tracking_freq" epoch
    stop = len(tracked_params['val_loss'])*tracked_params['tracking_freq']
    x2 = np.arange(0, stop, tracked_params['tracking_freq'])
    axs[0].plot(x2, tracked_params['val_loss'], label='val_loss')
    axs[0].legend(loc=1)
    
    # add the learning rate to the plot
    ax2 = axs[0].twinx()
    ax2.plot(tracked_params['lr_list'], label='learning_rate', color='red')
    ax2.set_ylabel('Learning Rate') 
    # fix that on this plot the legends are overlapping
    
    # plot the training accuracy
    axs[1].plot(x2,tracked_params['train_acc_top1'], label='train_acc')
    # plot the validation accuracy
    axs[1].plot(x2,tracked_params['val_acc_top1'], label='val_acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    #axs[1].set_title(desc)
    axs[1].legend()
    # add some spacing between plots 
    if save: 
        #plt.savefig(f'./plots/{name}.png')
        fig.savefig(save_path+'.png')
    if plot:    
        plt.show()
    
    # free up memory
    fig.clear()
    plt.close(fig)
    # add the val_loss to the plot

def plot_trainings(tracked_params1, tracked_params2, name1, name2):
    # Plot the training curves for two tracked_params on the same plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot for train_loss and val_loss
    desc1 = f"{name1}\nEpochs: {tracked_params1['epochs']}, Weight Decay: {tracked_params1['weight_decay']}, Learning Rate: {tracked_params1['lr']}, Momentum: {tracked_params1['momentum']}, Reduce Factor: {tracked_params1['reduce_factor']}"
    desc2 = f"{name2}\nEpochs: {tracked_params2['epochs']}, Weight Decay: {tracked_params2['weight_decay']}, Learning Rate: {tracked_params2['lr']}, Momentum: {tracked_params2['momentum']}, Reduce Factor: {tracked_params2['reduce_factor']}"
    
    # Plot train_loss and val_loss for model 1
    x1 = np.arange(0, len(tracked_params1['train_loss']), 1)
    axs[0].plot(x1, tracked_params1['train_loss'], label=f'train_loss - {name1}')
    axs[0].plot(x1, tracked_params1['val_loss'], label=f'val_loss - {name1}')
    
    # Plot train_loss and val_loss for model 2
    x2 = np.arange(0, len(tracked_params2['train_loss']), 1)
    axs[0].plot(x2, tracked_params2['train_loss'], label=f'train_loss - {name2}')
    axs[0].plot(x2, tracked_params2['val_loss'], label=f'val_loss - {name2}')
    
    print('plotting train loss: 1:',tracked_params1["train_loss"],'2:',tracked_params2["train_loss"])
    print('plotting val loss: 1:',tracked_params1["val_loss"],'2:',tracked_params2["val_loss"])

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss and Validation Loss')
    axs[0].legend()
    
    # Plot for train_acc and val_acc
    axs[1].plot(x1, tracked_params1['train_acc_top1'], label=f'train_acc - {name1}')
    axs[1].plot(x1, tracked_params1['val_acc_top1'], label=f'val_acc - {name1}')
    axs[1].plot(x2, tracked_params2['train_acc_top1'], label=f'train_acc - {name2}')
    axs[1].plot(x2, tracked_params2['val_acc_top1'], label=f'val_acc - {name2}')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training Accuracy and Validation Accuracy')
    axs[1].legend()
    
    plt.suptitle(f"Comparison of Training Curves for {name1} and {name2}")
    
    # save the plot
    plt.savefig(f'./plots/{name1}_{name2}.png')
    #plt.show()
    # free up memory
    fig.clear()
    plt.close(fig)

def plot_trainings_mean_min_max(tracked_params_dict,display_train_acc,display_only_mean,save,save_path,display): 
    # dict is of the form:
    # {"model_name": tracked_params(mean,min,,max), ...}
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # Plot for train_loss and val_loss for each model
    colors = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","magenta","yellow","black","darkblue","darkorange","darkgreen","darkred","darkpurple","darkbrown","darkpink","darkgray","darkolive","darkcyan","darkmagenta","darkyellow","darkblack"]
    # reverse colors array
    colors = colors[::-1]
    for model_name, (p_mean,p_min,p_max) in tracked_params_dict.items():
        color = colors.pop()
        if display_train_acc:
            train_color = colors.pop()
        # plot the mean values: 
        x1 = np.arange(1, len(p_mean['train_loss'])+1, 1)
        axs.plot(x1,p_mean['val_acc_top1'], label=f'val acc - {model_name}',color=color)
        if display_train_acc:
            axs.plot(x1,p_mean['train_acc_top1'], label=f'train acc - {model_name}',color=train_color)
        
        
        # also plot min and max data: 
        if not display_only_mean:
            # plot val_acc min,max range
            axs.fill_between(x1, p_min['val_acc_top1'], p_max['val_acc_top1'], alpha=0.2,color=color)

            if display_train_acc: 
            # plot train_acc min,max range
                axs.fill_between(x1, p_min['train_acc_top1'], p_max['train_acc_top1'], alpha=0.2,color=train_color)
            
    # add legend and titles to the plot
    axs.legend()
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Accuracy')
    if display_train_acc:
        axs.set_title('Training Accuracy and Validation Accuracy')
    else:
        axs.set_title('Validation Accuracy')
        
    # save the image to disk
    if save:
        fig.savefig(save_path+'.png')

    if display:
        plt.show()
        plt.close(fig)

def list_tracked_params_to_avg(list_tracked_params, also_min_max=False):
    if not also_min_max:
        # Compute the average for each dimension in tracked_params
        avg_tracked_params = {}
        # for all keys except the individual losses and accuracies just copy the ones, since they are all the same
        for key in list_tracked_params[0].keys():
            avg_tracked_params[key] = list_tracked_params[0][key]
    
        keys = ['train_loss','val_loss','train_acc_top1','train_acc_top5','val_acc_top1','val_acc_top5',"lr"]
        for key in keys:
            avg_tracked_params[key] = np.mean([params[key] for params in list_tracked_params], axis=0)
        return avg_tracked_params
    else: 
        avg_tracked_params = {}
        min_tracked_params = {}
        max_tracked_params = {}
        for key in list_tracked_params[0].keys():
            avg_tracked_params[key] = list_tracked_params[0][key]
            min_tracked_params[key] = list_tracked_params[0][key]
            max_tracked_params[key] = list_tracked_params[0][key]
        keys = ['train_loss','val_loss','train_acc_top1','train_acc_top5','val_acc_top1','val_acc_top5',"lr"]
        for key in keys:
            avg_tracked_params[key] = np.mean([params[key] for params in list_tracked_params], axis=0)
            min_tracked_params[key] = np.min([params[key] for params in list_tracked_params], axis=0)
            max_tracked_params[key] = np.max([params[key] for params in list_tracked_params], axis=0)
        return avg_tracked_params, min_tracked_params, max_tracked_params

def plot_training_avg(list_tracked_params,name,plot=True, save=False):
    avg_tracked_params = list_tracked_params_to_avg(list_tracked_params)
    # Visualize the average tracked_params using the plot_training function
    plot_training(avg_tracked_params, name,plot,save)


def train(model, loaders, lr=0.1, epochs=100, momentum=0.9, weight_decay=0.0001, reduce_patience=5, reduce_factor=0.2, tracking_freq=5,early_stopping_patience=10, early_stopping_min_epochs=100, do_tracking=True, verbose=False,device="cuda",verbose_tqdm=True,optimizer="None"):
    # dictionary to keep track of training params and results
    if verbose: print("starting training")
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
    train_dict['optimizer'] = optimizer
    if optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = SGD(model.parameters(),momentum=momentum, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("optimizer not supported")
    
    criterion = ch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=reduce_patience, verbose=verbose, factor=reduce_factor)
    len_train_loader = len(loaders['train'])
    len_val_loader = len(loaders['test'])

    best_val_acc = 0
    early_stopping_counter = 0
    model = model.to(device)
    for i in tqdm(range(epochs),disable= not verbose_tqdm):
        model.train()
        running_loss = 0.0
        total_correct, total_num, total_correct_top5 = 0., 0., 0.

        for ims, labs in loaders['train']:
            ims = ims.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            #with autocast():
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
                    val_ims = val_ims.to(device, non_blocking=True)
                    val_labs = val_labs.to(device, non_blocking=True)
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
