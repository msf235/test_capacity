
"""Script for running cnn capacity experiments.

Sets of parameters used for running simulations can be found in the file 
cnn_capacity_params.py.

Datasets and the relevant group-shifted versions of datasets can be found
in datasets.py."""

import os
import sys
import inspect
import math
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
from sklearn import svm, linear_model
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')
from joblib import Parallel, delayed
import pickle as pkl
import numpy as np
import warnings
from typing import *

import timm
import models
import model_output_manager as mom
import cnn_capacity_params as cp
import datasets
import cnn_capacity_utils as utils

output_dir = 'output'
fig_dir = 'figs'
# rerun = True # If True, rerun the simulation even if a matching simulation is
               # found saved to disk
rerun = False
# n_cores = 40  # Number of processor cores to use for multiprocessing. Recommend
# n_cores = 20  # Number of processor cores to use for multiprocessing. Recommend
# n_cores = 15
# n_cores = 20
n_cores = 10
# n_cores = 5
# n_cores = 1 # setting to 1 for debugging.
# seeds = [3, 4, 5, 6, 7]
seeds = [3, 4, 5]

## Collect parameter sets in a list of dictionaries so that simulations can be
## automatically saved and loaded based on the values in the dictionaries.
# param_set_name = 'random_1d_conv_exps'
# param_set_name = 'random_1d_conv_exps'
# param_set_name = 'random_2d_conv_exps'
# param_set_name = 'random_2d_conv_shift2_exps'
# param_set_name = 'vgg11_cifar10_exps'
# param_set_name = 'vgg11_cifar10_circular_exps'
# param_set_name = 'vgg11_cifar10_efficient_exps'

# param_set_names = ['randpoint_exps']
# param_set_names = ['randpoint_efficient_exps']
# param_set_names = ['vgg11_cifar10_efficient_exps', 'vgg11_cifar10_gpool_exps']
# param_set_names = ['vgg11_cifar10_efficient_exps']
# param_set_names = ['vgg11_cifar10_circular_exps']
param_set_names = ['vgg11_cifar10_gpool_exps']
# param_set_names = ['random_2d_conv_exps', 'random_2d_conv_gpool_exps']
# param_set_names = ['random_2d_conv_gpool_exps']
# param_set_names = ['random_2d_conv_efficient_exps']
# param_set_names = ['vgg11_cifar10_exps']
print('Running {}'.format('  '.join(param_set_names)))

param_set = []
for name in param_set_names:
    param_set += cp.param_sets[name]

# ImageNet directory
image_net_dir = '/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC/val'
# image_net_dir = '/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC/val'

# class HingeLoss(torch.nn.Module):
    # def __init__(self):
        # super().__init__()

    # def forward(self, output, target):
        # hinge_loss = 1 - torch.mul(output, target)
        # hinge_loss = torch.relu(hinge_loss)
        # # hinge_loss[hinge_loss < 0] = 0
        # return hinge_loss.mean()

def hinge_loss(output, target):
    hinge_loss = 1 - output*target[:, np.newaxis]
    hinge_loss[hinge_loss < 0] = 0
    return hinge_loss.mean()


# % Main function for capacity. This function is memoized based on its
# parameters.
def get_capacity(
    n_channels, n_inputs, seed=3, n_dichotomies=100, max_epochs=500,
    max_epochs_no_imp=None, improve_tol=1e-3, batch_size=256, img_size_x=10,
    img_size_y=10, img_channels=3, net_style='rand_conv', layer_idx=0,
    dataset_name='gaussianrandom', shift_style='2d', shift_x=1, shift_y=1,
    pool_over_group=False, perceptron_style='standard',
    pool=None, pool_x=None, pool_y=None,
    fit_intercept=True, center_response=True):
    """Take number of channels of response (n_channels) and number of input
    responses (n_inputs) and a set of hyperparameters and return the capacity
    of the representation.

    This function checks to see how large the dataset and representation
    would be in memory. If this value, times n_cores, is <= 30GB, then
    the code calls linear_model.LinearSVC on the entire dataset.
    If not, the code calls linear_model.SGDClassifier on batches of
    the data.

    Parameters
    ----------
    n_channels : int
		Number of channels in the network response.
    n_inputs : int
		Number of input samples to use.
    n_dichotomies : int
		Number of random dichotomies to test
    max_epochs : int
		Maximum number of epochs. This is also the max number of iterations
        when using LinearSVC.
    # max_epochs_no_imp : int
        Not implemented. Training will stop after this number of epochs without
        improvement
    # improve_tol : 
		Not implemented. The tolerance for improvement.
    batch_size : Optional[int]
		Batch size. If None, this is set to the size of the dataset.
    img_size_x : int
		Size of image x dimension.
    img_size_y : int
		Size of image y dimension.
    net_style : str 
        Style of network. Valid options are 'vgg11', 'alexnet', 'effnet', 'grid', 
        'rand_conv', and 'randpoints'.
    layer_idx : int
        Index for layer to get from conv net.
    dataset_name : str 
		The dataset. Options are 'imagenet', 'cifar10', and 'gaussianrandom'
    shift_style : str 
        Shift style. Options are 1d (shift in only x dimension) and 2d (use
        input shifts in both x and y dimensions).
    shift_x : int
		Number of pixels by which to shift in the x direction
    shift_y : int
		Number of pixels by which to shift in the y direction
    pool_over_group : bool 
        Whether or not to average (pool) the representation over the group
        before fitting the linear classifier.
    perceptron_style : str {'efficient', 'standard'}
        How to train the output weights. If 'efficient' then use the trick
        of finding a separating hyperplane for the identity group operations
        and then applying the average group projector to this hyperplane.
    pool : Optional[str] 
		Pooling to use for representation. Options are None, 'max', and 'mean'.
        Only currently implemented for net_style='rand_conv'.
    pool_x : Optional[int] 
		Size in pixels of pool in x direction. Set to None if pool is None.
    pool_y : Optional[int] 
		Size in pixels of pool in y direction. Set to None if pool is None.
    fit_intercept : bool 
		Whether or not to fit the intercept in the linear classifier.
        This currently throws an error when set to True since I haven't
        got the intercept working with perceptron_style = 'efficient' yet.
    center_response : bool 
        Whether or not to mean center each representation response.
    seed : int
		Random number generator seed. Currently haven't guaranteed perfect
        reproducibility.
    """
    if pool is None:
        pool_x = None
        pool_y = None
    if max_epochs_no_imp is None:
        improve_tol = None
    loc = locals()
    args = inspect.getfullargspec(get_capacity)[0]
    params = {arg: loc[arg] for arg in args}
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if mom.run_exists(params, output_dir) and not rerun: # Memoization
        run_id = mom.get_run_entry(params, output_dir)
        run_dir = output_dir + f'/run_{run_id}'
        try:
            with open(run_dir + '/get_capacity.pkl', 'rb') as fid:
                savedict = pkl.load(fid)
            return savedict['capacity']
        except FileNotFoundError:
            pass

    if perceptron_style == 'efficient':
        if pool_over_group:
            raise AttributeError("""perceptron_style=efficient not implemented
                                with group pooling.""")
    if fit_intercept:
        raise AttributeError("fit_intercept=True not currently implemented.")
    torch.manual_seed(seed)

    ## Someday I may choose to use feature hooks.
    ## For now I avoid it so I have the power to only do inference
    ## in the network only up to the needed layer. However, this 
    ## requires that the network model be modified to have a get_features method.
    # def register_feature_hooks(net, k, hook_fn):
        # cnt = 0
        # for name, layer in net._modules.items():
            # #If it is a sequential, don't register a hook on it
            # # but recursively register hook on all it's module children
            # if isinstance(layer, nn.Sequential):
                # register_feature_hooks(layer, k)
            # else: # it's a non sequential. Register a hook
                # layer.register_forward_hook(hook_fn)
                # cnt = cnt + 1
                # if cnt >= k:
                    # break
    pool_efficient_shift = 0

    if net_style[:5] == 'vgg11':
        if net_style[6:] == 'circular':
            net = models.vgg('vgg11_bn', 'A', batch_norm=True, pretrained=True,
                            circular_conv=True)
        else:
            net = models.vgg('vgg11_bn', 'A', batch_norm=True, pretrained=True)
        net.eval()
        def feature_fn(inputs):
            with torch.no_grad():
                feats = net.get_features(inputs, layer_idx)
                feats = feats[:, :n_channels]
                return feats
        if perceptron_style == 'efficient' or pool_over_group:
            if layer_idx > 2:
                if layer_idx > 6:
                    raise AttributeError("""This parameter combination not
                                         supported.""")
                pool_efficient_shift = 1

    elif net_style == 'alexnet':
        net = models.alexnet(pretrained=True)
        net.eval()
        def feature_fn(inputs):
            with torch.no_grad():
                feats = net.get_features(inputs)
                feats = feats['conv_layers'][layer_idx][:, :n_channels]
                return feats
    elif net_style == 'effnet':
        net = timm.models.factory.create_model('efficientnet_b2', pretrained=True)
        net.eval()
        def feature_fn(input):
            with torch.no_grad():
                feats = net.get_features(input)[layer_idx]
                feats = feats[:, :n_channels]
                return feats
    elif net_style == 'grid':
        convlayer = torch.nn.Conv2d(3, n_channels, 4, bias=False)
        torch.nn.init.xavier_normal_(convlayer.weight)
        # torch.nn.init.normal_(convlayer.weight)
        net = torch.nn.Sequential(
            convlayer,
            torch.nn.ReLU(),
            models.MultiplePeriodicAggregate2D(((14, 14), (8, 8))),
        )
        net.eval()
        def feature_fn(input):
            with torch.no_grad():
                hlist = net(input)
            hlist = [h.reshape(*h.shape[:2], -1) for h in hlist]
            h = torch.cat(hlist, dim=-1)
            return h
    elif net_style == 'rand_conv':
        convlayer = torch.nn.Conv2d(img_channels, n_channels,
                                    (img_size_x, img_size_y),
                            padding='same', padding_mode='circular',
                            bias=False)
        torch.nn.init.xavier_normal_(convlayer.weight)
        # torch.nn.init.normal_(convlayer.weight)
        # torch.nn.init.orthogonal_(convlayer.weight)
        layers = [convlayer, torch.nn.ReLU()]
        if pool is not None:
            if pool == 'max':
                pool_layer = torch.nn.MaxPool2d((pool_x, pool_y),
                                                (pool_x, pool_y)) 
            elif pool == 'mean':
                pool_layer = torch.nn.AvgPool2d((pool_x, pool_y),
                                                (pool_x, pool_y)) 
            if pool is not None or pool_over_group:
                if layer_idx > 1:
                    pool_efficient_shift = 1
            layers.append(pool_layer)
        layers = layers[:layer_idx+1]
        net = torch.nn.Sequential(*layers)
        net.eval()
        def feature_fn(input):
            with torch.no_grad():
                h = net(input)
                if center_response:
                    hflat = h.reshape(*h.shape[:2], -1)
                    hmean = hflat.mean(dim=-1, keepdim=True)
                    hms = hflat - hmean
                    hms_rs = hms.reshape(*h.shape)
                    return hms_rs
                return h
    elif net_style == 'randpoints':
        net = torch.nn.Module()
        net.eval()
        def feature_fn(inputs):
            return inputs
    else:
        raise AttributeError('net_style option not recognized')
    if net_style == 'randpoints':
        inp_channels = n_channels
    else:
        inp_channels = img_channels
    if dataset_name.lower() == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        # transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
        # ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((img_size_x, img_size_y)), #224 is typical
            transforms.ToTensor(),
            normalize,
        ])
        
        img_dataset = torchvision.datasets.ImageFolder(root=image_net_dir,
                                                        transform=transform_test)
        random_samples = torch.randperm(len(img_dataset))[:n_inputs]
        core_dataset = datasets.SubsampledData(img_dataset, random_samples)
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                              std=(0.2023, 0.1994, 0.2010))])
        img_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
            transform=transform)
        random_samples = torch.randperm(len(img_dataset))[:n_inputs]
        core_dataset = datasets.SubsampledData(img_dataset, random_samples)
    elif dataset_name.lower() == 'gaussianrandom':
        def zero_one_to_pm_one(y):
            return 2*y - 1
        core_dataset = datasets.FakeData(n_inputs,
                            (inp_channels, img_size_x, img_size_y),
                            target_transform=zero_one_to_pm_one)
    else:
        raise AttributeError('dataset_name option not recognized')

    if n_cores == 1:
        num_workers = 4
    else:
        num_workers = 0

    # if pool_over_group:
        # dataset = core_dataset
    if shift_style == '1d':
        dataset = datasets.ShiftDataset1D(core_dataset, shift_y)
    elif shift_style == '2d':
        dataset = datasets.ShiftDataset2D(core_dataset, shift_x, shift_y)
    else:
        raise AttributeError('Unrecognized option for shift_style.')
    if perceptron_style == 'efficient' or pool_over_group:
        datasetfull = dataset
        dataset = core_dataset
        inputsfull = torch.stack([x[0] for x in datasetfull])
        coreidxfull = [x[2] for x in datasetfull]
        dataloaderfull = torch.utils.data.DataLoader(
            datasetfull, batch_size=batch_size, num_workers=num_workers,
            shuffle=False)

    if batch_size is None or batch_size == len(dataset):
        batch_size = len(dataset)
        inputs = torch.stack([x[0] for x in dataset])
        if pool_over_group or perceptron_style == 'efficient':
            core_idx = list(range(len(dataset)))
        else:
            core_idx = [x[2] for x in dataset]
        dataloader = [(inputs, None, core_idx)]
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True)



    test_input, test_label = next(iter(dataloader))[:2]
    # plt.figure(); plt.imshow(dataset[100][0].transpose(0,2).transpose(0,1)); plt.show()
    h_test = feature_fn(test_input)
    if h_test.shape[1] < n_channels:
        raise AttributeError("""Error: network response produces fewer channels
                             than n_channels.""")
    N = torch.prod(torch.tensor(h_test.shape[2:])).item()

    P = utils.compute_pi_mean_reduced_2D(h_test.shape[-2], h_test.shape[-1],
                                   shift_x, shift_y) 
    Pt = P.T.copy()
    # P = np.ones((1, N)) / N
    # Pt = np.ones((N, 1)) / N
    # Pfullt = np.ones((N, N)) / N
    
    # # %%  Test data sampling
    # ds = dataloader.dataset
    # def no(k): Get network output
        # return net(ds[k][0].unsqueeze(dim=0)).squeeze()
    
    # cnt = 0
    # for input, label, core_idx in dataloader:
        # cnt += 1
        # print(input.shape)
        # print(label)
        # print(core_idx)
        # print(cnt)

    # # %% 

    # loss_fn = HingeLoss()
    loss_fn = hinge_loss

    # def score(w, X, Y):
        # Ytilde = X @ w
        # return np.mean(np.sign(Ytilde) == np.sign(Y))
    
    def score(w, X, Y):
        Ytilde = X @ w
        return np.mean(np.sign(Ytilde) == np.sign(Y-.5))
        # return np.mean(np.sign(Ytilde-.5) == np.sign(Y-.5))

    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def log_regscore(w, X, Y):
        Ytilde = X @ w
        ps = sigmoid(Ytilde)
        return np.mean(np.sign(ps-.5) == np.sign(Y-.5))

    def class_acc(outs, targets):
        correct = 1.0*(outs * random_labels > 0)
        return torch.mean(correct)

    def dich_loop(process_id=None):
        """Generates random labels and returns the accuracy of a classifier
        trained on the dataset."""
        np.random.seed(seed)
        rndseed = np.random.randint(10000)
        torch.manual_seed(process_id + rndseed)  # Be very careful with this
        # np.random.seed(process_id+rndseed) 
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
        while len(set(class_random_labels.tolist())) < 2:
            class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
        # perceptron = linear_model.SGDClassifier(
            # tol=1e-18, alpha=1e-16, fit_intercept=fit_intercept,
            # max_iter=max_epochs)
        perceptron = linear_model.LogisticRegression(
            tol=1e-18, C=1e8, fit_intercept=fit_intercept,
            max_iter=max_epochs, random_state=seed + rndseed)
        # perceptron = svm.LinearSVC(
            # tol=1e-18, C=1e8, fit_intercept=fit_intercept,
            # max_iter=max_epochs, random_state=seed)
        if batch_size == len(dataset):
            inputs = dataloader[0][0]
            core_idx = dataloader[0][2]
            h = feature_fn(inputs)
            if perceptron_style == 'efficient' or pool_over_group:
                if perceptron_style == 'efficient':
                    hfull = feature_fn(inputsfull)
                    Xfull = hfull.reshape(hfull.shape[0], -1).numpy()
                    Yfull = class_random_labels[coreidxfull].numpy()
                    Yfull = (Yfull + 1)/2
                Y = np.array(class_random_labels)
                if pool_efficient_shift == 1:
                    inputs21 = torch.roll(inputs, shifts=1, dims=-2) 
                    h21 = feature_fn(inputs21)
                    inputs12 = torch.roll(inputs, shifts=1, dims=-1) 
                    h12 = feature_fn(inputs12)
                    inputs22 = torch.roll(inputs21, shifts=1, dims=-1) 
                    h22 = feature_fn(inputs22)
                    h = torch.cat((h, h21, h12, h22), dim=0)
                    Y = np.concatenate((Y,Y,Y,Y), axis=0)
                hrs = h.reshape(*h.shape[:2], -1)
                centroids = hrs @ Pt
                X = centroids.reshape(centroids.shape[0], -1).numpy()
            else:
                X = h.reshape(h.shape[0], -1).numpy()
                Y = class_random_labels[core_idx].numpy()
            Y = (Y + 1)/2
            perceptron.fit(X, Y)
            if perceptron_style == 'efficient':
                wtemp = perceptron.coef_.copy()
                wtemp = wtemp.T @ P
                wtemp = wtemp.reshape(-1)
                curr_avg_acc = score(wtemp, Xfull, Yfull)
            else:
                curr_avg_acc = perceptron.score(X, Y)

            # curr_best_loss = 1000
            # not_improved_cntr = 0
            # for epoch in range(max_epochs):
                # perceptron.partial_fit(X, Y, classes=(-1, 1))
                # if perceptron_style == 'efficient':
                    # wtemp = perceptron.coef_.copy()
                    # wtemp = wtemp.T @ P
                    # wtemp = wtemp.reshape(-1)
                    # curr_avg_acc = score(wtemp, Xfull, Yfull)
                # else:
                    # curr_avg_acc = perceptron.score(X, Y).item()
                # if curr_avg_acc == 1.0:
                    # break
                # if max_epochs_no_imp is not None:
                    # loss = loss_fn(X, Y)
                    # if curr_best_loss >= loss - improve_tol:
                        # not_improved_cntr += 1
                    # else:
                        # not_improved_cntr = 0
                    # curr_best_loss = min(curr_best_loss, loss)
                    # if not_improved_cntr >= max_epochs_no_imp:
                        # break
        else:
            for epoch in range(max_epochs):
                losses_epoch = []
                class_acc_epoch = []
                for k1, data in enumerate(dataloader):
                    inputs = data[0]
                    h = feature_fn(inputs)
                    if pool_over_group:
                        hrs = h.reshape(*h.shape[:2], -1)
                        centroids = hrs @ Pt
                        # centroids = centroids.unique(dim=0)
                        X = centroids.reshape(centroids.shape[0], -1).numpy()
                        Y = np.array(class_random_labels)
                    else:
                        core_idx_batch = data[2]
                        X = h.reshape(h.shape[0], -1).numpy()
                        Y = class_random_labels[core_idx_batch].numpy()
                    perceptron.partial_fit(X, Y, classes=(-1, 1))
                    class_acc_epoch.append(perceptron.score(X, Y).item())
                    curr_avg_acc = sum(class_acc_epoch)/len(class_acc_epoch)
                if perceptron_style == 'efficient':
                    for k1, data in enumerate(dataloaderfull):
                        core_idx_batch = data[2]
                        hfull = feature_fn(data[0])
                        Xfull = hfull.reshape(hfull.shape[0], -1).numpy()
                        Yfull = class_random_labels[core_idx_batch].numpy()
                        wtemp = perceptron.coef_.copy()
                        wtemp = wtemp.T @ P
                        wtemp = wtemp.reshape(-1)
                        curr_avg_acc = score(wtemp, Xfull, Yfull)

                    # perc_compl = round(100*(k2/len(dataloader)))
                    # if process_id is None:
                        # print(f'Epoch {epoch} progress {perc_compl}%', end='\r')
                    # else:
                        # print(f'Process {process_id}: Epoch {epoch} progress {perc_compl}%', end='\r')
                    # print(f'Process {process_id}: Epoch average acc {curr_avg_acc}')
                # if process_id is None:
                    # print(f'Epoch {epoch}/{max_epochs}', end='\r')
                # else:
                    # print(f'Process {process_id}: Epoch {epoch}/{max_epochs}', end='\r')
                # if curr_avg_loss >= curr_best_loss - improve_tol:
                    # num_no_imp += 1
                # else:
                    # num_no_imp = 0
                # curr_best_loss = min(curr_avg_loss, curr_best_loss)
                # if num_no_imp > max_epochs_no_imp:
                    # break
                if curr_avg_acc == 1.0:
                    break
        return curr_avg_acc

            ## Debug code for computing centroids directly
            # core_idx = torch.tensor(core_idx)
            # centroids_inp = torch.zeros(n_inputs, *input.shape[1:])
            # for k2 in range(n_inputs):
                # centroids_inp[k2] = torch.mean(input[core_idx==k2], dim=0)
            # centroids = torch.zeros(n_inputs, *h.shape[1:])
            # Yc = torch.zeros(n_inputs)
            # for k2 in range(n_inputs):
                # centroids[k2] = torch.mean(h[core_idx==k2], dim=0)
                # Yc[k2] = Y[core_idx==k2][0]
            # centroids_inp_f = centroids_inp.reshape(*centroids_inp.shape[:2], -1)
            # C_inp = centroids_inp_f[:,0].T @ centroids_inp_f[:,0]
            # ew_inp, ev_inp = np.linalg.eigh(C_inp)
            # centroids_f = centroids.reshape(*centroids.shape[:2], -1)
            # C = centroids_f[:,0].T @ centroids_f[:,0]
            # ew, ev = np.linalg.eigh(C)
            # centroids_f_rs = centroids_f.reshape(centroids_f.shape[0], -1)
            # C = centroids_f_rs.T @ centroids_f_rs
            # ew, ev = np.linalg.eigh(C)
            # fitter = svm.LinearSVC(tol=1e-12, max_iter=40000, C=30.,
                                  # fit_intercept=fit_intercept)
            # fitter.fit(centroids_f_rs, Yc)
            # acc = fitter.score(centroids_f_rs, Yc)

            # fitter = svm.LinearSVC(tol=1e-18, C=1e16, fit_intercept=fit_intercept,
                                   # max_iter=max_epochs)
            ## Debug code for checking rank of data 
            # Xmc = X - np.mean(X, axis=0)
            # C = X.T @ Xmc
            # ew, ev = np.linalg.eigh(C)
            # fitter.fit(X, Y)
            # print(Y)
            # ax.scatter(X[:,0], X[:,1], c=Y)
            # plt.show()


    if n_cores > 1:
        print(f"Beginning parallelized loop over {n_dichotomies} dichotomies.")
        class_acc_dichs = Parallel(n_jobs=n_cores, batch_size=1, verbose=10)(
            delayed(dich_loop)(k1) for k1 in range(n_dichotomies))
    else:
        print(f"Beginning serial loop over {n_dichotomies} dichotomies.")
        class_acc_dichs = []
        for k1 in range(n_dichotomies):
            class_acc_dichs.append(dich_loop(k1))
            print(f'Finished dichotomy: {k1+1}/{n_dichotomies}', end='\r')

    capacity = (1.0*(torch.tensor(class_acc_dichs) == 1.0)).mean().item()
    if fit_intercept:
        alpha = n_inputs / (n_channels + 1)
    else:
        alpha = n_inputs / n_channels
    print(f'alpha: {round(alpha,5)}, capacity: {round(capacity,5)}')
    ## Now save results of the run to a pickled dictionary
    run_id = mom.get_run_entry(params, output_dir)
    run_dir = output_dir + f'/run_{run_id}/'
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir + 'get_capacity.pkl', 'wb') as fid:
        savedict = pkl.dump({'capacity': capacity}, fid)

    return capacity

def cover_theorem(P, N):
    frac_dich = 0
    for k in range(min(P,N)):
        frac_dich += math.factorial(P-1) / math.factorial(P-1-k) / math.factorial(k)
    frac_dich = 2**(1-1.0*P) * frac_dich
    return frac_dich

## Run script by calling get_capacity 
if __name__ == '__main__':
    plot_vars = ['n_channels', 'n_inputs', 'layer_idx']

    results_table = pd.DataFrame()
    for seed in seeds:
        for i0, params in enumerate(param_set):
            print(f"Starting param set {i0}/{len(param_set)} with seed {seed}")
            capacity = get_capacity(seed=seed, **params)
            layer = params['layer_idx']
            n_input = params['n_inputs']
            n_channel = params['n_channels']
            net_style = params['net_style']
            offset = int(params['fit_intercept'])
            pool_over_group = params['pool_over_group']
            alpha = n_input / (n_channel + offset)
            cover_capacity = cover_theorem(n_input, n_channel)
            pool_over_group = params['pool_over_group']
            d1 = {'seed': seed, 'alpha': alpha, 'n_inputs': n_input,
                  'n_channels': n_channel, 'n_channels_offset':
                  n_channel + offset, 'fit_intercept': params['fit_intercept'],
                  'layer': layer, 'net_style': net_style, 'capacity': capacity,
                  'pool_over_group': pool_over_group
                 }
            # for var in plot_vars:
                # d1[var] = params[var]
            d1 = pd.DataFrame(d1, index=[0])
            results_table = results_table.append(d1, ignore_index=True)

    for catcol in ('layer',):
        results_table[catcol] = results_table[catcol].astype('category')

    if len(results_table['net_style'].unique()) > 1:
        style = 'net_style'
    else:
        style = None
    style = 'pool_over_group'
    os.makedirs('figs', exist_ok=True)
    results_table.to_pickle('figs/most_recent.pkl')
    alpha_table = results_table.drop(
        columns=['n_channels', 'n_inputs', 'n_channels_offset',
                 'fit_intercept'])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.lineplot(ax=ax, x='alpha', y='capacity', data=alpha_table,
                 hue='layer', style=style)
    # sns.boxplot(ax=ax, x='alpha', y='capacity', data=alpha_table,
                 # hue=style)
    # sns.lineplot(ax=ax, x='alpha', y='capacity', data=alpha_table,
                 # hue=style)
    nmin = results_table['n_channels_offset'].min()
    nmax = results_table['n_channels_offset'].max()
    pmin = results_table['n_inputs'].min()
    pmax = results_table['n_inputs'].max()
    alphamin = results_table['alpha'].min()
    alphamax = results_table['alpha'].max()
    cover_cap = {p/n: cover_theorem(p, n) for n in range(nmin, nmax+1)
                for p in range(pmin, pmax+1) if alphamin <= p/n <= alphamax}
    cover_cap_maxpool = {p/n: cover_theorem(2*p, n) for n in range(nmin, nmax+1)
                for p in range(pmin, pmax+1) if alphamin <= p/n <= alphamax}
    ax.plot(list(cover_cap.keys()), list(cover_cap.values()), linestyle='dotted',
           color='blue', label='theory')
    # ax.plot(list(cover_cap_maxpool.keys()), list(cover_cap_maxpool.values()), linestyle='dotted',
           # color='red', label='theory maxpool')
    ax.legend()
    ax.set_ylim([-.01, 1.01])
    figname = '__'.join(param_set_names)
    fig.savefig(f'figs/{figname}.pdf', bbox_inches='tight')

