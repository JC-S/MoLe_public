'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': ['D', 64, 'M', 128, 'D', 128, 'M', 256, 'D', 256, 'D', 256, 'M', 512, 'D', 512, 'D', 512, 'M', 512, 'D', 512, 'D', 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGG(nn.Module):
    def __init__(self, vgg_name, dropout=False, num_classes=10):
        super(VGG, self).__init__()
        self.dropout = dropout
        self.first_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.features = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 64
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                if self.dropout:
                    layers += [nn.Dropout2d(self.dropout)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.BatchNorm2d(x)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.layers = layers
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
