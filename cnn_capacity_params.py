import torch

# Random CNN layer
random_2d_conv = dict(
n_dichotomies = 100, # Number of random dichotomies to test
n_inputs = [40], # Number of input samples to test
alphas = torch.linspace(0.8, 3.0, 10),
# n_inputs = [16], # Number of input samples to test
max_epochs = 500, # Maximum number of epochs.
# max_epochs_no_imp = 100, # Not implemented. Training will stop after
                           # this number of epochs without improvement
# improve_tol = 1e-3, # Not implemented. The tolerance for improvement.
batch_size = 256, # Batch size if training with SGD
img_size_x = 10, # Size of image x dimension.
img_size_y = 10, # Size of image y dimension.
# img_size_x = 224, # Size of image x dimension.
# img_size_y = 224, # Size of image y dimension.
# net_style = 'effnet', # Efficientnet layers.
# net_style = 'grid', # Not fully implemented. Grid cell CNN.
net_style = 'rand_conv', # Random convolutional layer.
# net_style = 'randpoints', # Random points. Used to make sure linear
                            # classifier is working alright.
layer_idx = [0], # Index for layer to get from conv net. Currently only
               # implemented for net_style = 'effnet'.
# dataset_name = 'imagenet', # Use imagenet inputs.
dataset_name = 'gaussianrandom', # Use Gaussian random inputs.
# shift_style = '1d', # Take input 1d shifts (shift in only x dimension).
shift_style = '2d', # Use input shifts in both x and y dimensions
shift_x = 1, # Number of pixels by which to shift in the x direction
shift_y = 1, # Number of pixels by which to shift in the y direction
# pool_over_group = True, # Whether or not to average (pool) the representation over the
pool_over_group = False, # group before fitting the linear classifier.
pool=None, # No maxpooling of the representation.
fit_intercept = True, # Whether or not to fit the intercept in the linear
                      # classifier
# fit_intercept = False,
# center_response = True, # Whether or not to mean center each representation
center_response = False,  # response 
)

# Random CNN layer with 2 pixel shifts
random_2d_conv_shift2 = random_2d_conv.copy()
random_2d_conv_shift2.update(
shift_x=2,
shift_y=2,
alphas=torch.linspace(3.0, 8.0, 10),
)

# Random CNN layer with max pooling
random_2d_conv_maxpool2 = random_2d_conv.copy()
random_2d_conv_maxpool2.update(
pool='max',
pool_x=2,
pool_y=2,
img_size_x=20,
img_size_y=20,
alphas=torch.linspace(.2, 1.5, 10),
)

# VGG11 on CIFAR10
vgg11_cifar10 = dict(
n_dichotomies = 100, # Number of random dichotomies to test
n_inputs = [12], # Number of input samples to test
alphas = torch.linspace(1.0, 3.0, 10),
layer_idx = [2, 6],
max_epochs = 500, # Maximum number of epochs.
batch_size = 512, # Batch size if training with SGD.
img_size_x = 32, # Size of image x dimension.
img_size_y = 32, # Size of image y dimension.
net_style = 'vgg11', # AlexNet layers.
dataset_name = 'cifar10', # Use imagenet inputs.
shift_style = '2d', # Use input shifts in both x and y dimensions
shift_x = 1, # Number of pixels by which to shift in the x direction
shift_y = 1, # Number of pixels by which to shift in the y direction
# pool_over_group = True, # Whether or not to average (pool) the representation over the
pool_over_group = False, # group before fitting the linear classifier.
pool=None,
fit_intercept = True, # Whether or not to fit the intercept in the linear
                      # classifier
# fit_intercept = False,
# center_response = True, # Whether or not to mean center each representation
center_response = False,  # response 
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
n_inputs = [12], # Number of input samples to test
alphas = torch.linspace(1.0, 3.0, 10),
max_epochs = 500, # Maximum number of epochs.
batch_size = 512, # Batch size if training with SGD.
img_size_x = 224, # Size of image x dimension.
img_size_y = 224, # Size of image y dimension.
# img_size_x = 64, # Size of image x dimension.
# img_size_y = 64, # Size of image y dimension.
net_style = 'effnet', # Efficientnet layers.
layer_idx = [0], # Index for layer to get from conv net.
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
