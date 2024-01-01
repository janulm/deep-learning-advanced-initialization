import infrastructure as inf


from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch    

import gc
import threading
import time

import numpy as np
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
from torchvision.transforms import v2
import torchvision
import torch.nn as nn
import torch.nn.functional as F

############## Hyperparameters ############## (Best values from Custom_ResNet18_Hyper_Param_Tuning.py)
lr = 0.01
optimizer = "SGD"
lr_reduce_patience = 5
normalization = True
epochs = 25


### best params that use SGD as optimizer: Best val acc:  57.23  for lr:  0.01  reduce_patience:  5  normalization:  True  optimizer:  SGD

device = inf.device
#print("Using device: ",device)
# import the model 
from torchvision.models import resnet18


def train_model(i, j):

    loaders = inf.get_loaders_cifar100_superclass_subsets_pytorch(i,j,batch_size=128,num_workers=3,normalize=normalization)
    model = resnet18(weights=None).to(device)
    model.fc = nn.Linear(512, 10).to(device)
    
    model, tracked_params = inf.train(model, loaders, epochs=epochs,lr=lr, momentum=0.9, tracking_freq=1, reduce_factor=0.5, reduce_patience=lr_reduce_patience, do_tracking=True, early_stopping_min_epochs=80, early_stopping_patience=5, verbose=False,device=device,optimizer=optimizer)        
    # store the model
    torch.save(model.state_dict(), f'./results_training_run2_Adams/models_sgd/model_{i}_{j}.pt')    
    # save the tracked params
    np.save(f"./results_training_run2_Adams/models_sgd/tracked_params{i}_{j}.npy", tracked_params)
    # print final accuracys
    #print(f"Final accuracys for superclasses {i} and {j}")
    print(f"Train accuracy: {tracked_params['train_acc_top1'][-1]}")
    print(f"Test accuracy: {tracked_params['val_acc_top1'][-1]}") 
    
    # once done remove the model, tracked params and loaders from storage
    del model, tracked_params, loaders

# write main function
if __name__ == "__main__":
    run = 0
    max_runs = 45
    for i in range(0, 10):
        for j in range(i + 1, 10):
            print(f"Starting run {run+1}/{max_runs}")
            train_model(i,j)
            run += 1
            