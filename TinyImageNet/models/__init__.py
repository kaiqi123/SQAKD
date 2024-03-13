"""
Refer to pytorch official website: https://github.com/pytorch/vision/tree/8e078971b8aebdeb1746fea58851e3754f103053/torchvision/models
"""

from .resnet_imagenet import resnet18_imagenet  #, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet110_cifar, resnet1202_cifar
from .mobilenet_v2 import mobilenet_v2
from .efficientnet import efficientnet_b0
from .vgg import vgg8, vgg8_bn, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn


model_dict = {
    'resnet18_imagenet': resnet18_imagenet,
    'resnet20_cifar': resnet20_cifar,
    'resnet32_cifar': resnet32_cifar,
    'resnet44_cifar': resnet44_cifar,
    'resnet56_cifar': resnet56_cifar,
    'resnet110_cifar': resnet110_cifar,
    'resnet1202_cifar': resnet1202_cifar,
    'mobilenet_v2': mobilenet_v2,
    "efficientnet_b0": efficientnet_b0,
    "vgg8": vgg8,
    "vgg8_bn": vgg8_bn,
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
}
