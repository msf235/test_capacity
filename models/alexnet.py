import torch
from torch import nn
import torchvision
from typing import *

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.ModuleList(( # Name to match torchvision model
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.ModuleList((
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = dict(conv_layers = [], avgpool=[], classifier=[])
        for layer in self.features:
            x = layer(x)
            features['conv_layers'].append(x)
        x = self.avgpool(x)
        features['avgpool'].append(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
            features['classifier'].append(x)
        return features
    
# class AlexNetCifar(nn.Module):
    # def __init__(self, num_classes: int = 1000):
        # super().__init__()
        # self.features = nn.ModuleList(( # Name to match torchvision model
            # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        # ))
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.ModuleList((
            # nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        # ))

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for layer in self.features:
            # x = layer(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # for layer in self.classifier:
            # x = layer(x)
        # return x

    # def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # features = dict(conv_layers = [], avgpool=[], classifier=[])
        # for layer in self.features:
            # x = layer(x)
            # features['conv_layers'].append(x)
        # x = self.avgpool(x)
        # features['avgpool'].append(x)
        # x = torch.flatten(x, 1)
        # for layer in self.classifier:
            # x = layer(x)
            # features['classifier'].append(x)
        # return features

def alexnet(pretrained: bool = False, progress: bool = True,
            dataset: str = 'imagenet',
            **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        # state_dict_torchvis = torch.hub.load_state_dict_from_url(model_urls['alexnet'],
                                                       # progress=progress)
        # state_dict = {}
        # for key, match_key in state_dict_match:
            # state_dict[key] = state_dict_torchvis[match_key]
        state_dict = torch.hub.load_state_dict_from_url(model_urls['alexnet'],
                                                       progress=progress)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    an = alexnet()
