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



device = inf.device
print("Using device: ",device)

# import the model 
from Custom_ResNet18 import custom_resnet_18


# do training on the models for the tupels of superclasses
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        
# Assuming custom_resnet_18 and inf are defined in other parts of your code

# Initialize a semaphore with the desired number of parallel threads
#max_threads = 4  # Adjust this number based on your system's capability
#thread_semaphore = threading.Semaphore(max_threads)

def train_model(i, j, epochs):
    # Acquire the semaphore
    #thread_semaphore.acquire()
    #print(f"Thread {i} and {j} aquired the semaphore")
    #try:
    print(f"Training model for superclasses {i} and {j}")
    paths = [f'./data/subsets/{dataset_name}_superclass_{i}_{j}.beton' for dataset_name in ["train", "test"]]
    loaders, start_time = inf.make_dataloaders_ffcv(paths[0], paths[1])
    model = custom_resnet_18(10)
    model = model.to(device)
    model, tracked_params = inf.train(model, loaders, epochs=epochs,lr=0.1, momentum=0.9, tracking_freq=2, reduce_factor=0.5, reduce_patience=5, do_tracking=True, early_stopping_min_epochs=80, early_stopping_patience=5, verbose=False)
    #print(f'Total time: {time.time() - start_time:.5f}')
    # store the model
    torch.save(model.state_dict(), f'./models/model_{i}_{j}_{epochs}.pt')    
    # save the tracked params
    np.save(f"./models/tracked_params{i}_{j}_{epochs}.npy", tracked_params)
    # print final accuracys
    #print(f"Final accuracys for superclasses {i} and {j}")
    print(f"Train accuracy: {tracked_params['train_acc_top1'][-1]}")
    print(f"Test accuracy: {tracked_params['val_acc_top1'][-1]}") 
    
    # once done remove the model, tracked params and loaders from storage
    del model, tracked_params, loaders, start_time
    torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection

    #print("Memory stats after freeing resources:")
    #print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    #print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    #print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    #finally:
        # Release the semaphore
       # thread_semaphore.release()

# Create a list to store the threads


for i in range(0, 20):
    for j in range(i + 1, 20):
        for e in [40,41,42]:
            train_model(i,j,e)
        
for i in range(0, 20):
    for j in range(i + 1, 20):
        for e in [40,41,42]:
            tracked_params = np.load(f"./models/tracked_params{i}_{j}_{e}.npy", allow_pickle=True).item()
            # plot training
            name = f'model_{i}_{j}_{e}'
            inf.plot_training(tracked_params,name, False, True)