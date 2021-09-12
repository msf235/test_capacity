import timm
import models
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
# from matplotlib import pyplot as plt

n_dichotomies = 20
n_inputs = 3
max_epochs = 100
max_epochs_no_imp = 20
improve_tol = 1e-3
batch_size = 256
num_channels = 5
# net_style = 'conv'
# net_style = 'grid'
net_style = 'rand_conv'
# dataset_name = 'imagenet'
dataset_name = 'random'
# shift_style = '1d'
shift_style = '2d'

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
    def feature_fn(input):
        with torch.no_grad():
            return net.get_features(input)[0]
elif net_style == 'grid':
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, num_channels, 3),
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
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, num_channels, 3, padding='same',
                        padding_mode='circular'),
        torch.nn.ReLU()
    )
    def feature_fn(input):
        with torch.no_grad():
            return net(input)
else:
    raise AttributeError('net_style option not recognized')
net.eval()
if dataset_name.lower() == 'imagenet':
    core_dataset = torchvision.datasets.ImageFolder(root=image_net_dir,
                                                    transform=transform_test)
elif dataset_name.lower() == 'random':
    core_dataset = UniformNoiseImages((10,10), n_inputs)
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
if shift_style == '1d':
    dataset = ShiftDataset1D(core_dataset, core_indices=random_samples)
elif shift_style == '2d':
    dataset = ShiftDataset2D(core_dataset, core_indices=random_samples)
else:
    raise AttributeError('Unrecognized option for shift_style.')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=2)
test_input, test_label, core_idx = next(iter(dataloader))
# plt.figure(); plt.imshow(dataset[100][0].transpose(0,2).transpose(0,1)); plt.show()
h_test = feature_fn(test_input)
N = torch.prod(torch.tensor(h_test.shape[1:]))

# # %%  Test data sampling
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


class Perceptron(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.readout_w = torch.nn.Parameter(torch.randn(N) / N**.5)

    def forward(self, input):
        return input @ self.readout_w

class_acc_dichs = []
for k1 in range(n_dichotomies):
    class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
    perceptron = Perceptron(N)
    optimizer = torch.optim.Adam(perceptron.parameters(), lr=.001,
                                weight_decay = 0*1e-10)
    curr_best_loss = 100.0
    num_no_imp = 0
    for epoch in range(max_epochs):
        losses_epoch = []
        class_acc_epoch = []
        for k2, (input, label, core_idx) in enumerate(dataloader):
            random_labels = class_random_labels[core_idx]
            h = feature_fn(input)
            h = h.reshape(h.shape[0], -1)
            optimizer.zero_grad()
            out = perceptron(h)
            # y = 2*label - 1
            loss = loss_fn(out, random_labels)
            loss.backward()
            optimizer.step()
            losses_epoch.append(loss.item())
            curr_avg_loss = sum(losses_epoch)/len(losses_epoch)
            class_acc_epoch.append(class_acc(out, random_labels).item())
            curr_avg_acc = sum(class_acc_epoch)/len(class_acc_epoch)
            perc_compl = round(100*(k2/len(dataloader)))
            print(f'Epoch {epoch} progress: {perc_compl}%')
            print(f'Current average loss: {curr_avg_loss}')
            print(f'Current average acc: {curr_avg_acc}')
            # print('\r')
        if curr_avg_loss >= curr_best_loss - improve_tol:
            num_no_imp += 1
        else:
            num_no_imp = 0
        curr_best_loss = min(curr_avg_loss, curr_best_loss)
        if num_no_imp > max_epochs_no_imp:
            break
        if curr_avg_acc == 1.0:
            break
    # Get classification accuracy
    class_acc_final = []
    for k2, (input, label, core_idx) in enumerate(dataloader):
        random_labels = class_random_labels[core_idx]
        h = feature_fn(input)
        h = h.reshape(h.shape[0], -1)
        out = perceptron(h)
        class_acc_final.append(class_acc(out, random_labels).item())
    class_acc_dichs.append(sum(class_acc_final) / len(class_acc_final))

capacity = (1.0*(torch.tensor(class_acc_dichs) == 1.0)).mean().item()
print(capacity)
