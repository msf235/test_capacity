import torch
import copy

def alphas_to_channels(alphas, n_inputs, fit_intercept):
    n_channels_temp = torch.round(n_inputs/alphas).int()
    n_channels_temp -= fit_intercept 
    n_channels_temp = n_channels_temp.tolist()
    n_channels = []
    [n_channels.append(x) for x in n_channels_temp if x not in n_channels]
    return n_channels

def exps_channels_and_layers(param_base, n_channels, layers=None):
    exps = []
    if layers is not None:
        for layer in layers:
            for n_channel in n_channels:
                temp = param_base.copy()
                temp['layer_idx'] = layer
                temp['n_channels'] = n_channel
                exps.append(temp)
    else:
        for n_channel in n_channels:
            temp = param_base.copy()
            temp['n_channels'] = n_channel
            exps.append(temp)
            
    return exps

param_sets = {}

## One-dimensional random CNN layer.
random_1d_conv = dict(
    n_dichotomies = 100, # Number of random dichotomies to test
    n_inputs = 40, # Number of input samples to test
    # n_inputs = 30, # Number of input samples to test
    max_epochs = 4000, # Maximum number of epochs.
    # max_epochs_no_imp = 200,
    # improve_tol = 1e-10
    batch_size = None, # Batch size if training with SGD
    img_size_x = 40, # Size of image x dimension.
    img_size_y = 1, # Size of image y dimension.
    net_style = 'rand_conv', # Random convolutional layer.
    # layer_idx = 0,
    # layer_idx = 1,
    # layer_idx = 2,
    dataset_name = 'gaussianrandom', # Use Gaussian random inputs.
    # img_channels = 100,
    img_channels = 3,
    # perceptron_style = 'efficient',
    shift_style = '2d', # Use input shifts in both x and y dimensions
    shift_x = 1, # Number of pixels by which to shift in the x direction
    shift_y = 1, # Number of pixels by which to shift in the y direction
    pool_over_group = False, # group before fitting the linear classifier.
    pool='max', 
    # pool=None,
    # pool_x = 2,
    # pool_y = 1,
    fit_intercept = False,
    center_response = False,
)
alphas = torch.linspace(0.5, 3.0, 15)
layer_idx = [1, 2]
# alphas = torch.linspace(1.5, 2.0, 2)
n_channels = alphas_to_channels(alphas, random_1d_conv['n_inputs'],
    int(random_1d_conv['fit_intercept']))
param_sets.update(
random_1d_conv_exps = exps_channels_and_layers(random_1d_conv, n_channels,
                                              layers=layer_idx)
)

## Random inputs
randpoint = random_1d_conv.copy()
randpoint['net_style'] = 'randpoints'
n_channels = alphas_to_channels(
    alphas, randpoint['n_inputs'], int(randpoint['fit_intercept']))
param_sets.update(
randpoint_exps = exps_channels_and_layers(randpoint, n_channels)
)
randpoint_efficient = randpoint.copy()
randpoint_efficient.update(
    perceptron_style = 'efficient'
)
param_sets.update(
randpoint_efficient_exps = exps_channels_and_layers(randpoint_efficient,
                                                    n_channels)
)

## Random CNN layer
random_2d_conv = random_1d_conv.copy()
random_2d_conv.update(
    perceptron_style='standard',
    img_size_x = 10,
    img_size_y = 10,
    pool_x = 2,
    pool_y = 2
)
alphas = torch.cat((torch.linspace(0.5, 1.0, 10), torch.linspace(1.0, 3.0, 10)),
                   dim=0)
layer_idx = [1, 2]
n_channels = alphas_to_channels(
    alphas, random_2d_conv['n_inputs'],
    int(random_2d_conv['fit_intercept']))
param_sets.update(
random_2d_conv_exps = exps_channels_and_layers(
    random_2d_conv, n_channels, layers=layer_idx)
)
random_2d_conv_gpool = random_2d_conv.copy()
random_2d_conv_gpool.update(
    pool_over_group=True,
)
param_sets.update(
    random_2d_conv_gpool_exps = exps_channels_and_layers(
            random_2d_conv_gpool, n_channels, layers=layer_idx)
)

random_2d_conv_efficient = random_2d_conv.copy()
random_2d_conv_efficient.update(
    perceptron_style='efficient',
)
param_sets.update(
random_2d_conv_efficient_exps = exps_channels_and_layers(
    random_2d_conv_efficient, n_channels, layers=layer_idx)
)

random_2d_conv_gpool = random_2d_conv.copy()
random_2d_conv_gpool.update(
    pool_over_group=True
)
random_2d_conv_gpool_exps = exps_channels_and_layers(
    random_2d_conv_gpool, n_channels, layers=layer_idx)

## Random CNN layer with 2 pixel shifts
random_2d_conv_shift2 = random_2d_conv.copy()
random_2d_conv_shift2.update(
    shift_x=2,
    shift_y=2,
)
alphas=torch.linspace(3.0, 8.0, 10)
n_channels = alphas_to_channels(
    alphas, random_2d_conv_shift2['n_inputs'],
    int(random_2d_conv_shift2['fit_intercept']))
param_sets.update(
random_2d_conv_shift2_exps = exps_channels_and_layers(
    random_2d_conv_shift2, n_channels)
)


## VGG11 on CIFAR10
vgg11_cifar10 = dict(
    n_dichotomies = 100, # Number of random dichotomies to test
    # n_dichotomies = 20, # Number of random dichotomies to test
    # n_dichotomies = 2, # Number of random dichotomies to test
    n_inputs = 20, # Number of input samples to test
    max_epochs = 500, # Maximum number of epochs.
    batch_size = None, # Batch size if training with SGD.
    img_size_x = 32, # Size of image x dimension.
    img_size_y = 32, # Size of image y dimension.
    net_style = 'vgg11',
    # net_style = 'vgg11_circular',
    # perceptron_style = 'efficient',
    dataset_name = 'cifar10', # Use imagenet inputs.
    shift_style = '2d', # Use input shifts in both x and y dimensions
    shift_x = 1, # Number of pixels by which to shift in the x direction
    shift_y = 1, # Number of pixels by which to shift in the y direction
    pool_over_group = False, # group before fitting the linear classifier.
    pool=None,
    fit_intercept = False, 
    center_response = False,  # response 
)
alphas = torch.linspace(0.5, 3.0, 15)
# alphas = torch.linspace(1.8, 2.2, 1)
layer_idx = [2, 3, 6]
# layer_idx = [3]
n_channels = alphas_to_channels(
    alphas, vgg11_cifar10['n_inputs'],
    int(vgg11_cifar10['fit_intercept']))
param_sets.update(
vgg11_cifar10_exps = exps_channels_and_layers(vgg11_cifar10, n_channels,
                                              layer_idx)
)
vgg11_cifar10_exps = exps_channels_and_layers(vgg11_cifar10, n_channels,
                                              layer_idx)

vgg11_cifar10_circular = vgg11_cifar10.copy()
vgg11_cifar10_circular.update(
    net_style='vgg11_circular',
)
param_sets.update(
vgg11_cifar10_circular_exps = exps_channels_and_layers(
    vgg11_cifar10_circular, n_channels, layer_idx)
)

vgg11_cifar10_efficient = vgg11_cifar10_circular.copy()
vgg11_cifar10_efficient.update(
    perceptron_style='efficient',
)
param_sets.update(
vgg11_cifar10_efficient_exps = exps_channels_and_layers(
    vgg11_cifar10_efficient, n_channels,
    # layer_idx[-2:-1],
    layer_idx,
)
)

vgg11_cifar10_gpool = vgg11_cifar10_circular.copy()
vgg11_cifar10_gpool.update(
    pool_over_group=True,
)
param_sets.update(
vgg11_cifar10_gpool_exps = exps_channels_and_layers(
    vgg11_cifar10_gpool, n_channels, layer_idx)
)

vgg11_cifar10_gpool_lay2 = vgg11_cifar10_gpool.copy()
vgg11_cifar10_gpool_lay2.update(
    layer_idx=2,
    n_inputs=20,
)
n_channels = alphas_to_channels(
    alphas, vgg11_cifar10_gpool_lay2['n_inputs'],
    int(vgg11_cifar10['fit_intercept']))
vgg11_cifar10_gpool_lay2_exps = exps_channels_and_layers(
    vgg11_cifar10_gpool_lay2, n_channels)

vgg11_cifar10_gpool_lay3 = vgg11_cifar10_gpool.copy()
vgg11_cifar10_gpool_lay3.update(
    layer_idx=3,
    n_inputs=20,
)
n_channels = alphas_to_channels(
    alphas, vgg11_cifar10_gpool_lay3['n_inputs'],
    int(vgg11_cifar10['fit_intercept']))
vgg11_cifar10_gpool_lay3_exps = exps_channels_and_layers(
    vgg11_cifar10_gpool_lay3, n_channels)

vgg11_cifar10_gpool_lay6 = vgg11_cifar10_gpool.copy()
vgg11_cifar10_gpool_lay6.update(
    layer_idx=6,
    n_inputs=20,
)
vgg11_cifar10_gpool_lay3_exps = exps_channels_and_layers(
    vgg11_cifar10_gpool_lay3, n_channels)
# vgg11_cifar10_gpool_exps = exps_channels_and_layers(
    # vgg11_cifar10_gpool, n_channels, layer_idx[-2:-1])


######### Not Used #############################
## Grid cell net
grid_2d_conv = dict(
    n_dichotomies = 100, # Number of random dichotomies to test
    n_inputs = 40, # Number of input samples to test
    max_epochs = 500, # Maximum number of epochs.
    # max_epochs_no_imp = 100, # Not implemented. Training will stop after
                               # this number of epochs without improvement
    # improve_tol = 1e-3, # Not implemented. The tolerance for improvement.
    batch_size = 256, # Batch size if training with SGD
    img_size_x = 56, # Size of image x dimension.
    img_size_y = 56, # Size of image y dimension.
    # img_size_x = 224, # Size of image x dimension.
    # img_size_y = 224, # Size of image y dimension.
    # net_style = 'effnet', # Efficientnet layers.
    net_style = 'grid', # Not fully implemented. Grid cell CNN.
    # net_style = 'rand_conv', # Random convolutional layer.
    # net_style = 'randpoints', # Random points. Used to make sure linear
                                # classifier is working alright.
    # dataset_name = 'imagenet', # Use imagenet inputs.
    dataset_name = 'gaussianrandom', # Use Gaussian random inputs.
    # shift_style = '1d', # Take input 1d shifts (shift in only x dimension).
    shift_style = '2d', # Use input shifts in both x and y dimensions
    shift_x = 1, # Number of pixels by which to shift in the x direction
    shift_y = 1, # Number of pixels by which to shift in the y direction
    # pool_over_group = True, # Whether or not to average (pool) the representation over the
    pool_over_group = False, # group before fitting the linear classifier.
    pool='max', 
    pool_x = 2,
    pool_y = 2,
    fit_intercept = True, # Whether or not to fit the intercept in the linear
                          # classifier
    # fit_intercept = False,
    # center_response = True, # Whether or not to mean center each representation
    center_response = False,  # response 
)
alphas = torch.linspace(0.8, 3.0, 15)
n_channels = alphas_to_channels(
    alphas, grid_2d_conv['n_inputs'],
    int(grid_2d_conv['fit_intercept']))
param_sets.update(
grid_2d_conv_exps=exps_channels_and_layers(grid_2d_conv_exps, n_channels,
                                           layer_idx)
)


# AlexNet on imagenet
alexnet_imagenet = dict(
n_dichotomies = 100, # Number of random dichotomies to test
n_inputs = [12], # Number of input samples to test
alphas = torch.linspace(1.0, 3.0, 10),
max_epochs = 500, # Maximum number of epochs.
batch_size = 512, # Batch size if training with SGD.
img_size_x = 224, # Size of image x dimension.
img_size_y = 224, # Size of image y dimension.
# img_size_x = 64, # Size of image x dimension.
# img_size_y = 64, # Size of image y dimension.
net_style = 'alexnet', # AlexNet layers.
layer_idx = [1], # Index for layer to get from conv net.
dataset_name = 'imagenet', # Use imagenet inputs.
shift_style = '2d', # Use input shifts in both x and y dimensions
shift_x = 14, # Number of pixels by which to shift in the x direction
shift_y = 14, # Number of pixels by which to shift in the y direction
# pool_over_group = True, # Whether or not to average (pool) the representation over the
pool_over_group = False, # group before fitting the linear classifier.
pool=None,
fit_intercept = True, # Whether or not to fit the intercept in the linear
                      # classifier
# fit_intercept = False,
# center_response = True, # Whether or not to mean center each representation
center_response = False,  # response 
)

# Efficientnet on imagenet
efficientnet_imagenet = dict(
n_dichotomies = 100, # Number of random dichotomies to test
n_inputs = 12, # Number of input samples to test
max_epochs = 500, # Maximum number of epochs.
batch_size = 512, # Batch size if training with SGD.
img_size_x = 224, # Size of image x dimension.
img_size_y = 224, # Size of image y dimension.
# img_size_x = 64, # Size of image x dimension.
# img_size_y = 64, # Size of image y dimension.
net_style = 'effnet', # Efficientnet layers.
layer_idx = 0, # Index for layer to get from conv net.
dataset_name = 'imagenet', # Use imagenet inputs.
shift_style = '2d', # Use input shifts in both x and y dimensions
shift_x = 14, # Number of pixels by which to shift in the x direction
shift_y = 14, # Number of pixels by which to shift in the y direction
# pool_over_group = True, # Whether or not to average (pool) the representation over the
pool_over_group = False, # group before fitting the linear classifier.
pool=None,
fit_intercept = True, # Whether or not to fit the intercept in the linear
                      # classifier
# fit_intercept = False,
# center_response = True, # Whether or not to mean center each representation
center_response = False,  # response 
)
alphas = torch.linspace(1.0, 3.0, 10)
n_channels = alphas_to_channels(
    alphas, efficientnet_imagenet['n_inputs'],
    int(efficientnet_imagenet['fit_intercept']))
efficientnet_imagenet_exps = []
for n_channel in n_channels:
    temp = efficientnet_imagenet.copy()
    temp['n_channels'] = n_channel
    efficientnet_imagenet_exps.append(temp)


