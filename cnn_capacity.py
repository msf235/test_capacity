import timm
from models import gridcellconv
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
# from matplotlib import pyplot as plt

n_dichotomies = 100
n_inputs = 1000
epochs = 5
batch_size = 124

# image_net_dir = '/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC/val'
image_net_dir = '/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC/val'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize,
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize,
])
def random_label(label):
    if torch.rand(1).item() < .5:
        return -1
    else:
        return 1

def get_shifted_img(img: torch.Tensor, gx: int, gy: int):
    img_ret = img.clone()
    img_ret = torch.roll(img_ret, (gx, gy), dims=(-2, -1))
    return img_ret

class ShiftDataset(torch.utils.data.Dataset):
    def __init__(self, core_dataset):
        self.core_dataset = core_dataset
        self.sx, self.sy = self.core_dataset[0][0].shape[1:]
        # self.targets = torch.tile(torch.tensor(self.core_dataset.targets),
                                  # (self.sx*self.sy,))
        # self.G = itertools.product((range(self.xs), range(sy))) # Lazy approach

    def __len__(self):
        return len(self.core_dataset)*self.sx*self.sy

    def __getitem__(self, idx):
        g_idx = idx % (self.sx*self.sy)
        idx_core = idx // (self.sx*self.sy)
        # g = self.G[g_idx] # Lazy approach
        gx = g_idx // self.sy 
        gy = g_idx % self.sy
        img, label = self.core_dataset[idx_core]
        return get_shifted_img(img, gx, gy), label

class HingeLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):

        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss.mean()


effnet = timm.models.factory.create_model('efficientnet_b2', pretrained=True)
core_dataset = torchvision.datasets.ImageFolder(root=image_net_dir,
                                                transform=transform_train)
class RandomLabels:
    def __init__(self, batch_size, seed):
        self.seed = seed
        torch.manual_seed(self.seed)
        self.batch_size = batch_size

    def reset(self):
        torch.manual_seed(self.seed)
    
    def get_random_labels(self):
        return 2*(torch.rand(self.batch_size) < 0.5) - 1

# core_dataset = timm.data.dataset_factory.create_dataset(
    # 'ImageFolder',
    # root='/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC',
    # batch_size=1)

                                         # sampler=binary_sampler)
# dataloader = timm.data.loader.create_loader(dataset, (3,224,224), 1,
                                           # num_workers=0,
                                            # persistent_workers=False,)
                                           # # sampler=binary_sampler)

dataset = ShiftDataset(core_dataset)
sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
    range(n_inputs))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=0,
                                        sampler=sub_sampler)
random_labels = RandomLabels(batch_size, 2)
test_input, test_label = next(iter(dataloader))
# plt.figure(); plt.imshow(dataset[100][0].transpose(0,2).transpose(0,1)); plt.show()
# %% 
# loss_fn = torch.nn.HingeEmbeddingLoss()
loss_fn = HingeLoss()

h_test = effnet.get_features(test_input)
h0 = h_test[0]
N = torch.prod(torch.tensor(h0.shape[1:]))

for k1 in range(n_dichotomies):
    # dataset = ShiftDataset(core_dataset)
    # binary_sampler = torch.utils.data.sampler.SubsetRandomSampler(binary_indices)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=124,
                                             # num_workers=0,
                                            # sampler=sub_sampler)
    class Perceptron(torch.nn.Module):
        def __init__(self, N):
            super().__init__()
            self.readout_w = torch.nn.Parameter(torch.randn(N) / N**.5)

        def forward(self, input):
            return input @ self.readout_w

    perceptron = Perceptron(N)

    optimizer = torch.optim.Adam(perceptron.parameters(), lr=.0001)
    for epoch in range(epochs):
        random_labels.reset()
        for input, label in dataloader:
            random_label_batch = random_labels.get_random_labels()
            h = effnet.get_features(input)[0]
            h = h.reshape(h.shape[0], -1)
            optimizer.zero_grad()
            out = perceptron(h)
            # y = 2*label - 1
            loss = loss_fn(out, random_label_batch)
            loss.backward()
            optimizer.step()
            print(loss.item())

