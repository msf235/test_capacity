import timm
import models
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
# from matplotlib import pyplot as plt

n_dichotomies = 100
n_inputs = 1000
epochs = 5
batch_size = 128
# net_style = 'conv'
net_style = 'grid'
dataset_name = 'imagenet'
# dataset_name = 'random'

image_net_dir = '/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC/val'
# image_net_dir = '/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC/val'

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

def get_shifted_img(img: torch.Tensor, gx: int, gy: int):
    img_ret = img.clone()
    img_ret = torch.roll(img_ret, (gx, gy), dims=(-2, -1))
    return img_ret

class UniformNoiseImages(torch.utils.data.Dataset):
    def __init__(self, img_size: tuple, dset_size: int):
        self.img_size = img_size
        self.dset_size = dset_size

    def __len__(self):
        return self.dset_size

    def __getitem__(self, idx):
        if idx >= self.dset_size:
            raise AttributeError('Index exceeds dataset size')
        torch.manual_seed(idx)
        img = torch.rand(3, *self.img_size)
        label = 2*(torch.rand(1).item() < .5) - 1
        return img, label
    

class ShiftDataset2D(torch.utils.data.Dataset):
    def __init__(self, core_dataset, core_indices=None):
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
    def __init__(self, core_dataset, core_indices=None):
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
        # g = self.G[g_idx] # Lazy approach
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

if net_style == 'conv':
    net = timm.models.factory.create_model('efficientnet_b2', pretrained=True)
elif net_style == 'grid':
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 10, 3),
        models.MultiplePeriodicAggregate2D(((14, 14), (8, 8)))
    )
net.eval()
if dataset_name.lower() == 'imagenet':
    core_dataset = torchvision.datasets.ImageFolder(root=image_net_dir,
                                                    transform=transform_test)
elif dataset_name.lower() == 'random':
    core_dataset = UniformNoiseImages((64,64), batch_size)
else:
    raise AttributeError('dataset_name option not recognized')
# core_dataset = timm.data.dataset_factory.create_dataset(
    # 'ImageFolder',
    # root='/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC',
    # batch_size=1)

                                         # sampler=binary_sampler)
# dataloader = timm.data.loader.create_loader(dataset, (3,224,224), 1,
                                           # num_workers=0,
                                            # persistent_workers=False,)
                                           # # sampler=binary_sampler)

num_samples_core = len(core_dataset)
random_samples = torch.randperm(num_samples_core)[:n_inputs]
dataset = ShiftDataset1D(core_dataset, core_indices=random_samples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=2)
test_input, test_label, core_idx = next(iter(dataloader))
# plt.figure(); plt.imshow(dataset[100][0].transpose(0,2).transpose(0,1)); plt.show()
class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1

if net_style == 'conv':
    with torch.no_grad():
        h_test = net.get_features(test_input)
    h0 = h_test[0]
    N = torch.prod(torch.tensor(h0.shape[1:]))
elif net_style == 'grid':
    with torch.no_grad():
        h_test = net(test_input)
    h_test = [h.reshape(*h.shape[:2], -1) for h in h_test]
    h_test = torch.cat(h_test, dim=-1)
    N = torch.prod(torch.tensor(h_test.shape[1:]))
else:
    raise AttributeError('net_style option not recognized')
    # N = h_test.shape[-1]

# # %%  Test data sampling
# cnt = 0
# for input, label, core_idx in dataloader:
    # cnt += 1
    # random_labels = class_random_labels[core_idx]
    # print(label)
    # print(core_idx)
    # print(random_labels)
    # print(cnt)

# %% 

loss_fn = HingeLoss()

class Perceptron(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.readout_w = torch.nn.Parameter(torch.randn(N) / N**.5)

    def forward(self, input):
        return input @ self.readout_w

for k1 in range(n_dichotomies):
    perceptron = Perceptron(N)
    optimizer = torch.optim.Adam(perceptron.parameters(), lr=.0001)
    for epoch in range(epochs):
        for input, label, core_idx in dataloader:
            random_labels = class_random_labels[core_idx]
            if net_style == 'conv':
                with torch.no_grad():
                    h = net.get_features(input)[0]
                h = h.reshape(h.shape[0], -1)
            elif net_style == 'grid':
                with torch.no_grad():
                    hlist = net(input)
                hlist = [h.reshape(*h.shape[:2], -1) for h in hlist]
                h = torch.cat(hlist, dim=-1)
            h = h.reshape(h.shape[0], -1)
            optimizer.zero_grad()
            out = perceptron(h)
            # y = 2*label - 1
            loss = loss_fn(out, random_labels)
            loss.backward()
            optimizer.step()
            print(loss.item())

