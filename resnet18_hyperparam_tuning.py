import numpy as np
import torch

import infrastructure as inf
from torchvision.models import resnet18

choices_lr = [0.1,0.01]
choices_normalization = [True ,False]
choices_lr_reduce_patience = [5,100] # the second setting disables reduction completely
choice_optimizer = ["SGD"] 

def main_train():
    device = inf.device
    print("Using device: ",device)

    torch.manual_seed(42)

    choices_tuples = [(2,4),(3,5),(4,6),(5,8,)]

    # Pytorch MPS enabled
    choices_dataloaders_normalized = [inf.get_loaders_cifar100_superclass_subsets_pytorch(i,j,batch_size=128,num_workers=3,normalize=True) for i,j in choices_tuples]
    choices_dataloaders_not_normalized = [inf.get_loaders_cifar100_superclass_subsets_pytorch(i,j,batch_size=128,num_workers=3,normalize=False) for i,j in choices_tuples]

    epochs = 30

    i = 0
    num_total_runs = len(choices_lr)*len(choices_lr_reduce_patience)*len(choices_normalization)*len(choice_optimizer)

    for lr in choices_lr:
        for reduce_patience in choices_lr_reduce_patience:
            for normalization in choices_normalization:
                for optimizer in choice_optimizer:
                    tracked_params = []
                    print("Starting run: ",i,"/",num_total_runs)
                    i += 1
                    # iterate over the choices_tuples and train the model
                    if normalization:
                        ### normalized data loaders
                        for loaders in choices_dataloaders_normalized:
                            
                            # generate a new random model for each run
                            model = resnet18(weights=None).to(device)

                            _, tracked_run =  inf.train(model, loaders, epochs=epochs,lr=lr, momentum=0.9, tracking_freq=1, reduce_factor=0.5, reduce_patience=reduce_patience, do_tracking=True, early_stopping_min_epochs=80, early_stopping_patience=5, verbose=False,device=device,optimizer=optimizer)        
                            tracked_params.append(tracked_run)

                        
                        # compute the average tracked params
                        avg_tracked = inf.list_tracked_params_to_avg(tracked_params)
                        # save the average tracked params to disk: 
                        name = f'results_training_run2_Adams/hyper_param_testing/tracked_params_{lr}_{reduce_patience}_{normalization}_{optimizer}.npy'
                        np.save(name,avg_tracked)
                        
                    else: 
                        # not normalized data loaders
                        for loaders in choices_dataloaders_not_normalized:
                            
                            # generate a new random model for each run
                            model = resnet18(pretrained=False).to(device)

                            trained_model, tracked_run =  inf.train(model, loaders, epochs=epochs,lr=lr, momentum=0.9, tracking_freq=1, reduce_factor=0.5, reduce_patience=5, do_tracking=True, early_stopping_min_epochs=80, early_stopping_patience=5, verbose=False,device=device,optimizer=optimizer)        
                            tracked_params.append(tracked_run)

                        
                        # compute the average tracked params
                        avg_tracked = inf.list_tracked_params_to_avg(tracked_params)
                        # save the average tracked params to disk: 
                        name = f'results_training_run2_Adams/hyper_param_testing/tracked_params_{lr}_{reduce_patience}_{normalization}_{optimizer}.npy'
                        np.save(name,avg_tracked)
                    
                    
                    
def main_plot(plot=True):
    # compute plots

    best_value = 0.0
    param_tuple = None

    for lr in choices_lr:
        for reduce_patience in choices_lr_reduce_patience:
            for normalization in choices_normalization:
                for optimizer in choice_optimizer:
                    ########### CREATING PLOTS FROM THE TRACKED PARAMAS
                    # read stored np array
                    name = f'results_training_run2_Adams/hyper_param_testing/tracked_params_{lr}_{reduce_patience}_{normalization}_{optimizer}'
                    tracked_params = np.load(name+".npy", allow_pickle=True).item()
                    
                    # compute the mean of the last 5 epochs on the validation acc
                    val = np.mean(tracked_params["val_acc_top1"][-5:])
                    if optimizer == "SGD" and val > best_value:
                        best_value = val
                        param_tuple = (lr,reduce_patience,normalization,optimizer)

                    # plot training
                    if plot:
                        inf.plot_training(tracked_params,name, False, True,name)
    

    print("Best val acc: ",best_value," for lr: ",param_tuple[0]," reduce_patience: ",param_tuple[1]," normalization: ",param_tuple[2]," optimizer: ",param_tuple[3])


#### main function call
                    
if __name__ == "__main__":
    main_plot(False)


    # the best hyperparameters are:
    # for lr:  0.01  reduce_patience:  5  normalization:  True  optimizer:  SGD