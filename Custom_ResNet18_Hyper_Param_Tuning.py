import infrastructure as inf


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

device = inf.device
print("Using device: ",device)

# import the model 
from Custom_ResNet18 import custom_resnet_18

from tqdm import tqdm

# do training on the models for the tupels of superclasses
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


i = 0
j = 1

choices_lr = [0.1]
choices_patience = [2,5,10]
choices_factor = [0.2,0.5,0.8]

for lr in choices_lr:
    for patience in choices_patience:
        for factor in choices_factor:
            ########## TRAINING 

            print(f"Training model for superclasses {i} and {j} lr = {lr} patience = {patience} factor = {factor}")
            paths =  [f'./data/subsets/{dataset_name}_superclass_{i}_{j}.beton' for dataset_name in ["train","test"]]
            loaders, start_time = inf.make_dataloaders_ffcv(paths[0],paths[1])
            model = custom_resnet_18(10)
            model  = model.to(device)
            model, tracked_params = inf.train(model, loaders,lr=lr,momentum=0.9,epochs=150,tracking_freq=2,reduce_factor=factor,reduce_patience=patience,do_tracking=True,early_stopping_min_epochs=150,early_stopping_patience=5,verbose=False)
            print(f'Total time: {time.time() - start_time:.5f}')
            # store the model   
            torch.save(model.state_dict(), f'./models/model_{i}_{j}.pt')	
            # save the tracked params
            np.save(f"./models/tracked_params{i}_{j}_{lr}_{patience}_{factor}.npy", tracked_params)
            
            # once done remove the model, tracked params and loaders from storage
            name = f'model_{i}_{j}'
            #inf.plot_training(tracked_params,name, False, True)
            del model, tracked_params, loaders, start_time
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
       
for lr in choices_lr:
    for patience in choices_patience:
        for factor in choices_factor:
                ########## TRAINING 
            ########### CREATING PLOTS FROM THE TRACKED PARAMAS
            # read stored np array
            tracked_params = np.load(f"./models/tracked_params{i}_{j}_{lr}_{patience}_{factor}.npy", allow_pickle=True).item()
            # plot training
            name = f'model_{i}_{j}_{lr}_{patience}_{factor}'
            inf.plot_training(tracked_params,name, False, True)
        
