import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys

from .custom_modules import QConv


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
        
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, args, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x, save_dict=None, lambda_dict=None):

        act_conv1 = x 
        out = F.relu(self.bn1(self.conv1(x)))
        act_conv2 = out 
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        # for distill
        self.out = out

        if save_dict:
            layer_num = save_dict["layer_num"]
            block_num = save_dict["block_num"]
            save_dict[f"layer{layer_num}.block{block_num}.conv1"] = act_conv1
            save_dict[f"layer{layer_num}.block{block_num}.conv2"] = act_conv2
        return out


class QBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, args, stride=1, option='A'):
        super(QBasicBlock, self).__init__()
        self.conv1 = QConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, args=args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, args=args)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     QConv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, args=args),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x, save_dict=None, lambda_dict=None):
        if save_dict:
            save_dict["conv_num"] = 1
        out = F.relu(self.bn1(self.conv1(x, save_dict, lambda_dict)))
        if save_dict:
            save_dict["conv_num"] = 2
        out = self.bn2(self.conv2(out, save_dict, lambda_dict)) 
        out += self.shortcut(x)
        out = F.relu(out)
        
        # for distill
        self.out = out
        
        return out