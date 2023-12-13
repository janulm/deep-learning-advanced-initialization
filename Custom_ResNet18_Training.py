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


# do training on the models for the tupels of superclasses
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        
# Assuming custom_resnet_18 and inf are defined in other parts of your code

# Initialize a semaphore with the desired number of parallel threads
max_threads = 1  # Adjust this number based on your system's capability
thread_semaphore = threading.Semaphore(max_threads)

def train_model(i, j):
    # Acquire the semaphore
    thread_semaphore.acquire()
    #print(f"Thread {i} and {j} aquired the semaphore")
    try:
        #print(f"Training model for superclasses {i} and {j}")
        paths = [f'./data/subsets/{dataset_name}_superclass_{i}_{j}.beton' for dataset_name in ["train", "test"]]
        loaders, start_time = inf.make_dataloaders(paths[0], paths[1])
        model = custom_resnet_18(10)
        model = model.to(device)
        model, tracked_params = inf.train(model, loaders, epochs=40,lr=0.1, momentum=0.9, tracking_freq=2, reduce_factor=0.5, reduce_patience=5, do_tracking=True, early_stopping_min_epochs=40, early_stopping_patience=5, verbose=False)
        #print(f'Total time: {time.time() - start_time:.5f}')
        # store the model
        torch.save(model.state_dict(), f'./models/model_{i}_{j}.pt')    
        # save the tracked params
        np.save(f"./models/tracked_params{i}_{j}.npy", tracked_params)
        
        # once done remove the model, tracked params and loaders from storage
        del model, tracked_params, loaders, start_time
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

        #print("Memory stats after freeing resources:")
        #print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        #print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        #print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    finally:
        # Release the semaphore
        thread_semaphore.release()

# Create a list to store the threads

"""
for i in range(0, 20):
    # this code then spawns 19-i threads for each i
    threads = []
    for j in range(i + 1, 20):
        # Create a thread for each model
        thread = threading.Thread(target=train_model, args=(i, j))
        threads.append(thread)
        thread.start()
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
"""

start_time = time.time()
# test on a single model
thread = threading.Thread(target=train_model, args=(0, 1))
thread.start()
thread.join()
print(f'Total time for 1 thread: {time.time() - start_time:.5f}')

# do test for 2 threads
start_time = time.time()
threads = []
for j in range(1, 3):
    # Create a thread for each model
    thread = threading.Thread(target=train_model, args=(0, j))
    threads.append(thread)
    thread.start()
# Wait for all threads to finish
for thread in threads:
    thread.join()
    
print(f'Total time for 2 threads: {time.time() - start_time:.5f}')

# do test for 4 threads
start_time = time.time()
threads = []
for j in range(1, 5):
    # Create a thread for each model
    thread = threading.Thread(target=train_model, args=(0, j))
    threads.append(thread)
    thread.start()
# Wait for all threads to finish
for thread in threads:
    thread.join()
    
print(f'Total time for 4 threads: {time.time() - start_time:.5f}')