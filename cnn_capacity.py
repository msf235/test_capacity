import timm
import torch
from torchvision.transforms import ToTensor

T = ToTensor()

effnet = timm.models.factory.create_model('efficientnet_b2', pretrained=True)

dataset = timm.data.dataset_factory.create_dataset(
    'ImageFolder',
    root='/home/matthew/datasets/imagenet/ILSVRC/Data/CLS-LOC',
    batch_size=1)

dataloader = timm.data.loader.create_loader(dataset, (3,224,224), 1)

img = next(iter(dataloader))[0]

def get_shifted_img(img: torch.Tensor, gx: int, gy: int):
    img_ret = img.clone()
    img_ret = torch.roll(img_ret, (gx, gy), dims=(-2, -1))

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

