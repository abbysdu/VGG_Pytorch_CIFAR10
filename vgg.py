import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import ImageDraw, ImageFont


cfg = {
    'VGG11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        
        self.feature = self.make_layers(cfg[vgg_name])
        
        self.classifier = nn.Sequential(
                        nn.Linear(512, 4096),
                        nn.ReLU(True),
                        nn.Linear(4096, 4096),
                        nn.ReLU(True),
                        nn.Linear(4096, 10))
        
    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(-1, 512))
        # pre = F.softmax(x, dim=1)  # nn.CrossEntropyLoss()里面包括了softmax，不用再进行softmax
        return x
