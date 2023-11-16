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
        

def make_dataloaders(train_dataset="./data/cifar_train.beton", val_dataset="./data/cifar_test.beton", batch_size=1024, num_workers=12):
    paths = {
        'train': train_dataset,
        'test': val_dataset

    }

    start_time = time.time()
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#file-cifar100_mean_std-py
    # this source details how and what mean and std of the datasets are
    # took the values from the source above and multiplied by 255
    # not sure this properly translates to std dev of the dataset TODO: check this	
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
    
    model = torchvision.models.resnet18(pretrained=False)
    # make fc a sequential layer
    model.fc = ch.nn.Sequential(ch.nn.Linear(model.fc.in_features, output_dim), ch.nn.Softmax(dim=1))
    model = model.to(device=device)
    return model


def train(model, loaders, lr=0.1, epochs=50, momentum=0.9, weight_decay=0.0001, lr_peak_epoch=5):
    
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    for _ in range(epochs):
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()


def evaluate(model, loaders, lr_tta=False):
    # lr_tta: whether to use test-time augmentation by flipping images horizontally
    model.eval()
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ims.flip(-1))
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')


load_cifar100()
loaders, start_time = make_dataloaders()
model = generate_model()
train(model, loaders)
print(f'Total time: {time.time() - start_time:.5f}')
evaluate(model, loaders)