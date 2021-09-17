import torch

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
# net_style = 'conv', # Efficientnet layers.
# net_style = 'grid', # Not fully implemented. Grid cell CNN.
net_style = 'rand_conv', # Random convolutional layer.
# net_style = 'randpoints', # Random points. Used to make sure linear
                            # classifier is working alright.
layer_idx = 0, # Index for layer to get from conv net. Currently only
               # implemented for net_style = 'conv'.
# dataset_name = 'imagenet', # Use imagenet inputs.
dataset_name = 'gaussianrandom', # Use Gaussian random inputs.
# shift_style = '1d', # Take input 1d shifts (shift in only x dimension).
shift_style = '2d', # Use input shifts in both x and y dimensions
shift_x = 1, # Number of pixels by which to shift in the x direction
shift_y = 1, # Number of pixels by which to shift in the y direction
# pool = True, # Whether or not to average (pool) the representation over the
pool = False , # group before fitting the linear classifier.
fit_intercept = True, # Whether or not to fit the intercept in the linear
                      # classifier
# fit_intercept = False,
# center_response = True, # Whether or not to mean center each representation
center_response = False,  # response 
)

random_2d_conv_shift2 = random_2d_conv.copy()
random_2d_conv_shift2['shift_x'] = 2
random_2d_conv_shift2['shift_y'] = 2

efficientnet_imagenet = dict(
n_dichotomies = 100, # Number of random dichotomies to test
n_inputs = [16], # Number of input samples to test
alphas = torch.linspace(3.0, 8.0, 10),
max_epochs = 500, # Maximum number of epochs.
batch_size = 256, # Batch size if training with SGD
img_size_x = 224, # Size of image x dimension.
img_size_y = 224, # Size of image y dimension.
net_style = 'conv', # Efficientnet layers.
layer_idx = 0, # Index for layer to get from conv net. Currently only
               # implemented for net_style = 'conv'.
dataset_name = 'imagenet', # Use imagenet inputs.
shift_style = '2d', # Use input shifts in both x and y dimensions
shift_x = 1, # Number of pixels by which to shift in the x direction
shift_y = 1, # Number of pixels by which to shift in the y direction
# pool = True, # Whether or not to average (pool) the representation over the
pool = False , # group before fitting the linear classifier.
fit_intercept = True, # Whether or not to fit the intercept in the linear
                      # classifier
# fit_intercept = False,
# center_response = True, # Whether or not to mean center each representation
center_response = False,  # response 
)
