# Deep Learning Advanced Initialization

Deep Learning Project ETHZ HS23 from Jannek Ulm, Leander Diaz-Bone, Alexander Bayer, and Dennis Jüni

We introduce a novel weight initialization technique for Convolutional Neural Networks, focusing on image classification using the CIFAR100 dataset. Our weight initialization technique initializes the convolutional filters by drawing samples from a clustering computed on previously learned filters. Our findings show that this method significantly outperforms known initialization techniques, improving validation accuracy and leading to better generalization on unseen image datasets. This offers a new approach to CNN weight initialization, with potential applications in more complex models and datasets, which may enable more efficient networks and faster training.

For more details please check out our [report](Leaning_an_Informed_CNN_Initialization_for_Image_Classification.pdf).



## Training the models

We provide two different python environments for training the models and running the experiments. The "environment_cuda" is working for us on a Linux Machine with CUDA 12, the "environment_mps" works on new Apple Silicon Macbooks and uses the Apples MPS GPU acceleration. For specific device/cuda versions one might need to adapt the environment files. When running the [training file](./resnet18_training.py) all the pre-trained models required for the experiments are automatically saved in the [models](./experiment_results/models) folder. 

## Running the experiments from the paper

Most experiments require pretrained models on subsets of CIFAR-100.
Due to the size of a pretrained model, they are not included in this repository, however they can easily be recalculated using [this](./Custom_ResNet18_Training.py) training file (for the original ResNet-18 model and also for the custom ResNet-18 model).
All models were trained on 10 classes, which were chosen from 2 superclasses with indices between 0 and 9.

## Rerunning the experiments

All experiments from the paper are shown in [final_experiments.ipynb](./final_experiments.ipynb).

1. Testing the accuracy a randomly initialized model on random tuples of superclasses with indices between 10 and 19.
2. Testing the accuracy of a pretrained model (on some superclasses with indices between 0 and 9) and then re-trained on a random tuple of superclasses with indices between 10 and 19.
3. Testing the initialization with Gabor filters with 1,2,6,10, and 17 layers being initialized.
4. Testing the number of pretrained models used for clustering (with 10 clusters and 17 layers being initialized) with Euclidean and Fourier distance.
5. Testing the number of clusters used for clustering (with 10 models and 17 layers being initialized) with Euclidean and Fourier distance.
6. Testing the number of layers initialized for clustering (with 10 models and 10 cllusters) with Euclidean and Fourier distance.
7. Testing a randomly initialized _custom_ ResNet-18.
8. Testing a _custom_ ResNet-18 which was clustered and permuted according to the alignment algorithm.
9. Testing a random ResNet-18 model on the Tiny ImageNet dataset.
10. Testing a pretrained ResNet-18 (pretrained on a subset of 10 CIFAR-100 superclasse) on the Tiny ImageNet dataset.
11. Testing a clustered ResNet-18 model on the Tiny ImageNet dataset.

All the validation and training accuracies during the run of these experiments were saved in [tracked_params](./experiment_results/tracked_params/).

## Plotting the results

All plots from the paper were generated using parameters in [tracked_params](./experiment_results/tracked_params/).
The specific code used can be found in [final_plotting.ipynb](./final_plotting.ipynb).

#### Figure 2 - Single-filter Clustering Initialization Methods

![Figure 2](/experiment_results/plots/figure2.png)

#### Figure 3 - Varying number of Clusters

![Figure 3](/experiment_results/plots/figure3.png)

#### Figure 4 - Varying number of layers

![Figure 4](/experiment_results/plots/figure4.png)

## Figure 5 - Varying number of models

![Figure 5](/experiment_results/plots/figure5.png)

## Figure 6 - Custom model with alignment

![Figure 6](/experiment_results/plots/figure6.png)

## Figure 7 - Gabor varying number of layers

![Figure 7](/experiment_results/plots/figure7.png)

## Figure 8 - Compare initializing first layer

![Figure 8](/experiment_results/plots/figure8.png)

## Figure 9 - Tiny ImageNet Comparison

![Figure 9](/experiment_results/plots/figure9.png)

## Table 1 - Tiny ImageNet accuracies

| Model                    | Epoch 5 | Epoch 15 |
| ------------------------ | ------- | -------- |
| Random initialization    | 28.246  | 30.924   |
| Pre-trained on CIFAR-100 | 28.338  | 30.268   |
| Clustered initialization | 31.1    | 35.374   |
