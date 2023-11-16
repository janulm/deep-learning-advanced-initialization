# deep-learning-advanced-initialization
Deep Learning Project ETHZ HS23

## Creating the python environment for ffcv: 

For CUDA devices: Clone the environment_cuda.yaml file
For Apple/MPS devices: TBD

Data: 

https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html

https://www.cs.toronto.edu/~kriz/cifar.html download link

download cifar 100 for python: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

CIFAR-100 -> 20 Superclasses, two vehicle classes extra for final model

We want to split classes: 



Model: ResNet-18, import using pytorch, is possible to initialize with pretrained and random weights.

ResnetPaper: https://arxiv.org/pdf/1512.03385.pdf



Datasets: 

100 classes in total, 20 superclasses, 5 classes per superclass

2 vehicle * 10 classes
18 something * 10 classes

for each superclass we have 5 classes


## Ways to split dataset to learn multiple models:
18 superclass: each 5 models

Way 1: 18 models: 5 classes classification per model
Way 2: 45 models: 2 classes classification per model

Way 3: for each 18 superclasses: (5 classes) 5 * 4/2 = 10 models: 2 classes classification per model
        combine every possible pair: 5*4/2 = 10 models
        => 180 models in total (Hopefully feasible), or sample from this space? 

Way 4: n*(n-1)/2 models: 2 classes classification per model
        combine every possible pair:
    4005 models in total (Infeasible)

Way 5: Way1 + Sampling 5 classes out of 90:
    maybe sampling such that each group consits of at most 3 superclasses? 


Layer1: 64 filters each 7*7: we could train (using way3) 180*64 filters -> 11520 filters 

## How do we get the initialization distribution?

For each layer (has n filters): gather from all trained models all filters: 
    cluster them into (m >= n)? clusters (some smart ways)

E.g. for layer 1 (64 filters) with way3: we have 11520 filters
    cluster them, 
    sample from the 64 highest clusters? 

## Experiment 1: 

Randomly intialize models
Test how good model works...

## Experiment 2: 

Use Gabor Filters as Initialization
Train models, get distribution? 

## Experiment 3:

Get distributions of each above 
and use them to initialize the model, compare which perform better on 2 vehicle superclasses?
In what architecture? 

We have the following initilization distributions to compare: 


random initialization:  (cheap)
pretrained and sampled using Gabor Filters as init: (expensive)
pretrained and sampled using random as init: (expensive)


Way 1: 10 class classification? 
Way 2: 2 class classification? -> test on 10*9/2 = 45 models

## TODO:

Find out how expensive it is to train 1 binary resnet-18 model