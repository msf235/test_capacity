import os
import timm
import models
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
from sklearn import svm, linear_model
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import model_output_manager as mom
import pickle as pkl
import numpy as np
import warnings
from typing import *
import fakedata
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

output_dir = 'output'
fig_dir = 'figs'
# rerun = True # If True, rerun the simulation even if a matching simulation is
               # found saved to disk
rerun = False
n_cores = 10
# n_cores = 1

n_dichotomies = 100 # Number of random dichotomies to test
n_inputs = [40] # Number of input samples to test
max_epochs = 500 # Maximum number of epochs if training with SGD
# max_epochs_no_imp = 100 # Not implemented. Training will stop after
                          # this number of epochs without improvement
# improve_tol = 1e-3 # Not implemented. The tolerance for improvement.
batch_size = 512 # Batch size if training with SGD
n_channels = list(range(10,60,2)) # Number of channels to test in output layer.
img_size_x = 20 # Size of image x dimension.
img_size_y = 1 # Size of image y dimension.
# net_style = 'conv' # Not fully implemented. Efficientnet layers.
# net_style = 'grid' # Not fully implemented. Grid cell CNN.
net_style = 'rand_conv' # Random convolutional layer.
# net_style = 'randpoints' # Random points. Used to make sure linear
                           # classifier is working alright.
# dataset_name = 'imagenet' # Not fully implemented. Use imagenet inputs.
dataset_name = 'gaussianrandom' # Use Gaussian random inputs.
# shift_style = '1d' # Take input 1d shifts (shift in only x dimension).
shift_style = '2d' # Use input shifts in both x and y dimensions
# pool = True # Whether or not to average (pool) the representation over the
              # group before fitting the linear classifier.
pool = False
fit_intercept = True # Whether or not to fit the intercept in the linear
                     # classifier
# fit_intercept = False
# center_response = True # Whether or not to mean center each representation
                         # response 
center_response = False
seed = 3 # RNG seed

# Collect hyperparameters in a dictionary so that simulations can be
# automatically saved and loaded based on the values.
hyperparams = dict(n_dichotomies=n_dichotomies, max_epochs=max_epochs,
                   batch_size=batch_size, 
                   img_size_x=img_size_x, img_size_y=img_size_y,
                   net_style=net_style,
                   dataset_name=dataset_name, shift_style=shift_style,
                   pool=pool, fit_intercept=fit_intercept,
                   center_response=center_response, seed=seed)


# ImageNet directory
image_net_dir = '/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC/val'
# image_net_dir = '/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC/val'

def get_shifted_img(img: torch.Tensor, gx: int, gy: int):
    """Receives input image img and returns a shifted version,
    where the shift in the x direction is given by gx and in the y direction
    by gy."""
    img_ret = img.clone()
    img_ret = torch.roll(img_ret, (gx, gy), dims=(-2, -1))
    return img_ret

class ShiftDataset2D(torch.utils.data.Dataset):
    """Takes in a normal dataset of images and produces a dataset that
    samples from 2d shifts of this dataset, keeping the label for each
    shifted version of an image the same."""
    def __init__(self, core_dataset, core_indices: Optional[list] = None):
        super().__init__()
        self.core_dataset = core_dataset
        self.core_indices = core_indices
        self.sx, self.sy = self.core_dataset[0][0].shape[1:]
        # self.targets = torch.tile(torch.tensor(self.core_dataset.targets),
                                  # (self.sx*self.sy,))  # Too large for memory
        # self.G = itertools.product((range(self.xs), range(sy))) # Lazy approach

    def __len__(self):
        return len(self.core_dataset)*self.sx*self.sy

    def __getitem__(self, idx):
        g_idx = idx % (self.sx*self.sy)
        idx_core = idx // (self.sx*self.sy)
        # g = self.G[g_idx] # Lazy approach
        gx = g_idx // self.sy 
        gy = g_idx % self.sy
        if self.core_indices is not None:
            idx_core = self.core_indices[idx_core]
        img, label = self.core_dataset[idx_core]
        return get_shifted_img(img, gx, gy), label, idx_core

class ShiftDataset1D(torch.utils.data.Dataset):
    """Takes in a normal dataset of images and produces a dataset that
    samples from 1d shifts of this dataset, keeping the label for each
    shifted version of an image the same."""
    def __init__(self, core_dataset, core_indices: Optional[list] = None):
        super().__init__()
        self.core_dataset = core_dataset
        self.core_indices = core_indices
        self.sy = self.core_dataset[0][0].shape[-1]
        # self.targets = torch.tile(torch.tensor(self.core_dataset.targets),
                                  # (self.sy,))

    def __len__(self):
        return len(self.core_dataset)*self.sy

    def __getitem__(self, idx):
        g_idx = idx % self.sy
        idx_core = idx // self.sy
        idx_core = idx // self.sy
        gy = g_idx % self.sy
        if self.core_indices is not None:
            idx_core = self.core_indices[idx_core]
        img, label = self.core_dataset[idx_core]
        return get_shifted_img(img, 0, gy), label, idx_core

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss.mean()

# % Main function for capacity. This function is memoized based on its
# parameters and the values in hyperparams.
def get_capacity(n_channels, n_inputs):
    """Take number of channels of response (n_channels) and number of input
    responses (n_inputs) and return the capacity of the representation"""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    params = hyperparams.copy()
    params.update({'n_channels': n_channels, 'n_inputs': n_inputs})
    if mom.run_exists(params, output_dir) and not rerun: # Memoization
        run_id = mom.get_run_entry(params, output_dir)
        run_dir = output_dir + f'/run_{run_id}'
        try:
            with open(run_dir + '/get_capacity.pkl', 'rb') as fid:
                savedict = pkl.load(fid)
            return savedict['capacity']
        except FileNotFoundError:
            pass

    if net_style == 'conv':
        net = timm.models.factory.create_model('efficientnet_b2', pretrained=True)
        def feature_fn(input):
            with torch.no_grad():
                return net.get_features(input)[0]
    elif net_style == 'grid':
        convlayer = torch.nn.Conv2d(3, n_channels, 4, bias=False)
        torch.nn.init.xavier_normal_(convlayer.weight)
        net = torch.nn.Sequential(
            convlayer,
            torch.nn.ReLU(),
            models.MultiplePeriodicAggregate2D(((14, 14), (8, 8))),
        )
        def feature_fn(input):
            with torch.no_grad():
                hlist = net(input)
            hlist = [h.reshape(*h.shape[:2], -1) for h in hlist]
            h = torch.cat(hlist, dim=-1)
            return h
    elif net_style == 'rand_conv':
        convlayer = torch.nn.Conv2d(3, n_channels, (img_size_x, img_size_y),
                            padding='same', padding_mode='circular',
                            bias=False)
        torch.nn.init.xavier_normal_(convlayer.weight)
        net = torch.nn.Sequential(
            convlayer,
            torch.nn.ReLU()
        )
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
        def feature_fn(input):
            return input
    else:
        raise AttributeError('net_style option not recognized')
    net.eval()
    if net_style == 'randpoints':
        inp_channels = n_channels
    else:
        inp_channels = 3
    if dataset_name.lower() == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        core_dataset = torchvision.datasets.ImageFolder(root=image_net_dir,
                                                        transform=transform_test)
    elif dataset_name.lower() == 'gaussianrandom':
        def zero_one_to_pm_one(y):
            return 2*y - 1
        core_dataset = fakedata.FakeData(n_inputs,
                            (inp_channels, img_size_x, img_size_y),
                            target_transform=zero_one_to_pm_one)
    else:
        raise AttributeError('dataset_name option not recognized')

    num_samples_core = len(core_dataset)
    random_samples = torch.randperm(num_samples_core)[:n_inputs]
    if shift_style == '1d':
        dataset = ShiftDataset1D(core_dataset,
                                 core_indices=random_samples.tolist())
    elif shift_style == '2d':
        dataset = ShiftDataset2D(core_dataset,
                                 core_indices=random_samples.tolist())
    elif shift_style == '2d_centroid':
        dataset = ShiftCentroid2D(core_dataset,
                                  core_indices=random_samples.tolist())
    else:
        raise AttributeError('Unrecognized option for shift_style.')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=0, shuffle=True)
    test_input, test_label, core_idx = next(iter(dataloader))
    # plt.figure(); plt.imshow(dataset[100][0].transpose(0,2).transpose(0,1)); plt.show()
    h_test = feature_fn(test_input)
    N = torch.prod(torch.tensor(h_test.shape[1:]))

    if len(dataloader.dataset) <= 7.5e08/n_cores: # Entire dataset should be
                                                  # <= 30 Gigabytes
        train_style = 'whole'
    else:
        train_style = 'batched'

    # # %%  Test data sampling
    # ds = dataloader.dataset
    # def no(k): Get network output
        # return net(ds[k][0].unsqueeze(dim=0)).squeeze()
    
    # cnt = 0
    # for input, label, core_idx in dataloader:
        # cnt += 1
        # random_labels = class_random_labels[core_idx]
        # print(input.shape)
        # print(label)
        # print(core_idx)
        # print(random_labels)
        # print(cnt)

    # # %% 

    loss_fn = HingeLoss()

    def class_acc(outs, targets):
        correct = 1.0*(outs * random_labels > 0)
        return torch.mean(correct)

    def dich_loop():
        """Generates random labels and returns the accuracy of a classifier
        trained on the dataset."""
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
        while len(set(class_random_labels.tolist())) < 2:
            class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
        perceptron = linear_model.SGDClassifier(fit_intercept=fit_intercept,
                                               alpha=1e-10)
        N = img_size_x * img_size_y
        Pt = torch.ones(N, 1)
        if train_style == 'batched':
            # print(f'Training using batched SGD')
            # curr_best_loss = 100.0
            # num_no_imp = 0
            for epoch in range(max_epochs):
                losses_epoch = []
                class_acc_epoch = []
                for k2, (input, label, core_idx) in enumerate(dataloader):
                    random_labels = class_random_labels[core_idx].numpy()
                    h = feature_fn(input)
                    if pool:
                        hrs = h.reshape(*h.shape[:2], -1)
                        centroids = hrs @ Pt
                        X = centroids.reshape(centroids.shape[0], -1).numpy().astype(float)
                        Y = np.array(class_random_labels)
                    else:
                        X = h.reshape(h.shape[0], -1).numpy()
                        Y = random_labels
                    perceptron.partial_fit(X, Y, classes=(-1, 1))
                    class_acc_epoch.append(perceptron.score(X, Y).item())
                    curr_avg_acc = sum(class_acc_epoch)/len(class_acc_epoch)
                    perc_compl = round(100*(k2/len(dataloader)))
                    print(f'Epoch {epoch} progress: {perc_compl}%')
                print(f'Epoch average acc: {curr_avg_acc}', end='\r\r')
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
        elif train_style == 'whole':
            # print('Training standard SVM.')
            if pool:
                ds = dataloader.dataset.core_dataset
                n = len(ds)
                inputs, labels = zip(*[ds[k] for k in range(n)])
                inputs = torch.stack(inputs)
                h = feature_fn(inputs)
                hrs = h.reshape(*h.shape[:2], -1)
                centroids = hrs @ Pt
                X = centroids.reshape(centroids.shape[0], -1).numpy().astype(float)
                Y = np.array(class_random_labels)
            else:
                n = len(dataloader.dataset)
                input, __, core_idx = zip(
                    *[dataloader.dataset[k] for k in range(n)])
                core_idx = list(core_idx)
                input = torch.stack(input)
                h = feature_fn(input)
                hnp = h.numpy().astype(float)
                X = hnp.reshape(h.shape[0], -1)
                Y = class_random_labels[core_idx].numpy().astype(float)

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

            fitter = svm.LinearSVC(tol=1e-12, max_iter=40000, C=30.,
                                  fit_intercept=fit_intercept)
            ## Debug code for checking rank of data 
            # Xmc = X - np.mean(X, axis=0)
            # C = X.T @ Xmc
            # ew, ev = np.linalg.eigh(C)
            fitter.fit(X, Y)
            acc = fitter.score(X,Y)
            # fig, ax = plt.subplots()
            # ax.scatter(X[:,0], X[:,1], c=Y)
            # plt.show()
            # print(acc)
            return acc


    if n_cores == 1:
        class_acc_dichs = [dich_loop() for k1 in range(n_dichotomies)]
    else:
        class_acc_dichs = Parallel(n_jobs=n_cores)(
            delayed(dich_loop)() for k1 in range(n_dichotomies))

    capacity = (1.0*(torch.tensor(class_acc_dichs) == 1.0)).mean().item()
    if fit_intercept:
        alpha = n_inputs / (n_channels + 1)
    else:
        alpha = n_inputs / n_channels
    print(f'alpha: {alpha}, capacity: {capacity}')
    ## Now save results of the run to a pickled dictionary
    run_id = mom.get_run_entry(params, output_dir)
    run_dir = output_dir + f'/run_{run_id}/'
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir + 'get_capacity.pkl', 'wb') as fid:
        savedict = pkl.dump({'capacity': capacity}, fid)

    return capacity

## Run script by calling get_capacity 
if __name__ == '__main__':
    param_list = list(itertools.product(n_channels, n_inputs))
    results_table = pd.DataFrame()
    # if run_mode == 'serial':
    torch.manual_seed(seed)
    for n_input in n_inputs:
        for n_channel in n_channels:
            if fit_intercept:
                alpha = n_input / (n_channel + 1)
            else:
                alpha = n_input / n_channel
            capacity = get_capacity(n_channel, n_input)
            d = {'n_channel': n_channel, 'n_input': n_input, 'alpha': alpha,
                 'capacity': capacity}
            results_table = results_table.append(d, ignore_index=True)
    # elif run_mode == 'parallel':
        # capacities = Parallel(n_jobs=4)(
            # delayed(get_capacity)(*p) for p in param_list)
        # for k1, (n_input, n_channel) in enumerate(param_list):
            # capacity = capacities[k1]
            # d = {'n_channel': n_channel, 'n_input': n_input, 'alpha': n_input /
                 # n_channel, 'capacity': capacity}
            # results_table = results_table.append(d, ignore_index=True)

    results_table.to_pickle('figs/most_recent.pkl')
    alpha_table = results_table.drop(columns=['n_channel', 'n_input'])
    fig, ax = plt.subplots()
    sns.lineplot(ax=ax, x='alpha', y='capacity', data=alpha_table)
    ax.set_ylim([-.01, 1.01])
    fig.savefig('figs/most_recent.pdf')

