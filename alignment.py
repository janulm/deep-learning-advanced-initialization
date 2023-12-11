import torch
import torchvision
import numpy as np
import copy
import sys
from Custom_ResNet18 import *

def permute_bias(permuted_model_bias, perm):
    permuted_model_bias.weight = torch.nn.Parameter(permuted_model_bias.weight[perm])
    permuted_model_bias.bias = torch.nn.Parameter(permuted_model_bias.bias[perm])
    permuted_model_bias.running_mean = permuted_model_bias.running_mean[perm]
    permuted_model_bias.running_var = permuted_model_bias.running_var[perm]

def permute_layer(permuted_layer, permuted_layer_next, permuted_bias, perm = None):
    if not perm is None: 
        permuted_layer.weight = torch.nn.Parameter(permuted_layer.weight[perm])
        permute_bias(permuted_bias, perm)
        permuted_layer_next.weight =  torch.nn.Parameter(permuted_layer_next.weight.transpose(0,1)[perm].transpose(0,1))

def permute_weights(model, perm1 = None, perm2 = None, perm3 = None, perm4 = None):
     with torch.no_grad():
        permuted_model = copy.deepcopy(model)
        # Permute conv1
        permute_layer(permuted_model.conv1, permuted_model.layer1[0].conv1,  permuted_model.bn1,           perm1)
        # Permute layer[0].conv1
        permute_layer(permuted_model.layer1[0].conv1, permuted_model.layer1[0].conv2, permuted_model.layer1[0].bn1, perm2)
        # Permute layer1[0].conv2
        permute_layer(permuted_model.layer1[0].conv2, permuted_model.layer1[1].conv1, permuted_model.layer1[0].bn2, perm3)
        # Permute layer1[1].conv1
        permute_layer(permuted_model.layer1[1].conv1, permuted_model.layer1[1].conv2, permuted_model.layer1[1].bn1, perm4)
        return permuted_model