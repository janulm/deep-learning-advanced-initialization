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
import torchvision

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

def load_cifar100(train_dataset="./data/cifar_train.beton", val_dataset="./data/cifar_test.beton"):
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
        

def make_dataloaders(train_dataset="./data/cifar_train.beton", val_dataset="./data/cifar_test.beton", batch_size=256, num_workers=12):
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
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device(device), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float32), # TODO check what the impact for float16 is (it was the initial value, and why it crashes with float16)	
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time


def generate_model(output_dim:int = 100):
    
    #ResNet general source: https://pytorch.org/vision/master/models/resnet.html
    
    model = torchvision.models.resnet18(weights=None)
    # make fc a sequential layer
    #model.fc = ch.nn.Sequential(ch.nn.Linear(model.fc.in_features, output_dim), ch.nn.Softmax(dim=1))
    model.fc = ch.nn.Linear(model.fc.in_features, output_dim)
    model = model.to(device=device)
    return model


def train(model, loaders, lr=0.1, epochs=100, momentum=0.9, weight_decay=0.0001,reduce_patience=5, reduce_factor=0.2,tracking_freq=5,do_tracking=True,verbose=True):
    
    # dictionary to keep track of training params and results
    train_dict = {}
    train_dict['lr'] = lr
    train_dict['epochs'] = epochs
    train_dict['momentum'] = momentum
    train_dict['weight_decay'] = weight_decay
    train_dict['reduce_patience'] = reduce_patience
    train_dict['reduce_factor'] = reduce_factor
    # results
    # training loss is tracked every epoch
    train_dict['train_loss'] = []
    
    # all other params are tracked every e.g. 10 epochs	(tracking_freq) if do_tracking is True
    train_dict['train_acc_top1'] = []
    train_dict['train_acc_top5'] = []
    train_dict['val_acc_top1'] = []
    train_dict['val_acc_top5'] = []
    
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = ch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=reduce_patience, verbose=True, factor=reduce_factor)
    len_train_loader = len(loaders['train'])
    
    for i in range(epochs):
        model.train()
        running_loss = 0.0
        #for ims, labs in tqdm(loaders['train']):
        for ims, labs in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = criterion(out, labs)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss)
        # save training loss
        print(f'Epoch {i+1}/{epochs}, Loss: {running_loss/len_train_loader}')
        train_dict['train_loss'].append(running_loss/len_train_loader)
        # keep track of other metrics
        if do_tracking and (i+1)%tracking_freq == 0:
            train_top1, train_top5, val_top1, val_top5 = evaluate(model, loaders, lr_tta=False,verbose=verbose)
            train_dict['train_acc_top1'].append(train_top1)
            train_dict['train_acc_top5'].append(train_top5)
            train_dict['val_acc_top1'].append(val_top1)
            train_dict['val_acc_top5'].append(val_top5)
    return model, train_dict

def evaluate(model, loaders, lr_tta=False,verbose=True):
    # lr_tta: whether to use test-time augmentation by flipping images horizontally
    model.eval()
    train_top1, train_top5, val_top1, val_top5 = 0., 0., 0., 0.
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num, total_correct_top5 = 0., 0., 0.
            for ims, labs in loaders[name]:
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ims.flip(-1))
                    # computing top1 accuracy
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
                    # computing top5 accuracy
                    total_correct_top5 += out.argsort(1)[:,-5:].eq(labs.unsqueeze(-1)).sum().cpu().item()
            if verbose:
                print(f'{name} (acc) top-1: {total_correct / total_num * 100:.1f}, top-5: {total_correct_top5 / total_num * 100:.1f} %')
            if name == 'train':
                train_top1, train_top5 = total_correct / total_num * 100, total_correct_top5 / total_num * 100
            else:
                val_top1, val_top5 = total_correct / total_num * 100, total_correct_top5 / total_num * 100
    return train_top1, train_top5, val_top1, val_top5

load_cifar100()
loaders, start_time = make_dataloaders(batch_size=256, num_workers=12)
model = generate_model()
# load model from checkpoint stored at ./models/model.pt
#model.load_state_dict(torch.load("./models/model.pt"))
model, tracked_params = train(model, loaders,epochs=300,tracking_freq=5,do_tracking=True,verbose=True)
print(f'Total time: {time.time() - start_time:.5f}')
evaluate(model, loaders)

# store the model   
torch.save(model.state_dict(), "./models/model.pt")	
# save the tracked params
np.save("./models/tracked_params.npy", tracked_params)