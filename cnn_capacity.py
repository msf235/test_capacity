import timm
from models import gridcellconv
import torch
import itertools
from torchvision.transforms import ToTensor

T = ToTensor()

def get_shifted_img(img: torch.Tensor, gx: int, gy: int):
    img_ret = img.clone()
    img_ret = torch.roll(img_ret, (gx, gy), dims=(-2, -1))

effnet = timm.models.factory.create_model('efficientnet_b2', pretrained=True)

core_dataset = timm.data.dataset_factory.create_dataset(
    'ImageFolder',
    root='/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC',
    batch_size=1)

class ShiftDataset(torch.utils.data.Dataset):
    def __init__(self, core_dataset, ):
        self.core_dataset = core_dataset
        img_s = self.core_dataset[0][0].shape
        self.sx = img_s[-2]
        self.sy = img_s[-1]
        # self.G = itertools.product((range(self.xs), range(sy))) # Lazy approach

    def __len__(self):
        return len(self.core_dataset)*self.sx*self.sy

    def __getitem__(self, idx):
        g_idx = idx % (self.sx*self.sy)
        idx_core = idx // (self.sx*self.sy)
        img = self.core_dataset[idx_core]
        # g = self.G[g_idx] # Lazy approach
        gx = g_idx // self.sy 
        gy = g_idx % self.sy
        img, label = self.core_dataset[idx_core]
        return get_shifted_img(img, gx, gy), label


dataset = ShiftDataset(core_dataset)
dataloader = timm.data.loader.create_loader(dataset, (3,224,224), 1)

img = next(iter(dataloader))[0]
# %% 

class Perceptron(torch.nn.Module):
    def __init__(self, N):
        self.readout_w = torch.nn.Parameter(torch.randn(N) / N**.5)

    def forward(self, input):
        return input @ self.readout_w

net = Perceptron(20)

epochs = 5
for epoch in range(epochs):
    for x, y in 



# batch = torch.stack([T(dataset_train[k][0]) forkk in range(2)])
# %% 
effnet.to('cuda')
h = effnet.get_features(img.to('cuda'))

def get_features_shifted(feature_function, dataloader, num_samples):
    img = next(iter(dataloader))[0]
    batch_size, channels, sx, sy = img.shape
    features = feature_function(img)
    features_full = [torch.zeros(num_samples, sx, sy, *feature.shape[1:])
                    for feature in features]
    features = []
    cnt = 0
    for img, label in dataloader:
        for gx in range(sx):
            for gy in range(sy):
                img_shifted = get_shifted_img(img, gx, gy)
                features_shifted = feature_function(img_shifted)

    

# pretrain_batchsize = 256
# dataset_pretrain = timm.data.dataset_factory.create_dataset(
    # 'ImageFolder',
    # root='/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC',
    # split=args.train_split, is_training=True,
    # batch_size=pretrain_batchsize)
# dataset_eval = timm.data.dataset_factory.create_dataset(
    # 'ImageFolder',
    # root='/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC',
    # split=args.val_split, is_training=False, batch_size=args.batch_size)

