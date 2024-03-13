'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

from .custom_modules import QConv


__all__ = ['vgg19_bn_fp', 'vgg19_bn_quant', 'vgg16_bn_fp', 'vgg16_bn_quant', 
            'vgg13_bn_fp', 'vgg13_bn_quant', 'vgg11_bn_fp', 'vgg11_bn_quant', 
            'vgg8_bn_fp', 'vgg8_bn_quant',
            ]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, args):
        super(VGGBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm

    def forward(self, x, save_dict=None, lambda_dict=None):
        out = self.conv2d(x)
        if self.batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        self.out = out
        return out


class QVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, args):
        super(QVGGBlock, self).__init__()
        self.conv2d = QConv(in_channels, out_channels, kernel_size=3, padding=1, args=args)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm

    def forward(self, x, save_dict=None, lambda_dict=None):
        if save_dict:
            save_dict["conv_num"] = 0
        out = self.conv2d(x, save_dict, lambda_dict)
        if self.batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        self.out = out
        return out


class MySequential(nn.Sequential):
    def forward(self, x, save_dict=None, lambda_dict=None):
        block_num = 0
        for module in self._modules.values():
            if save_dict:
                save_dict["block_num"] = block_num
            x = module(x, save_dict, lambda_dict)
            block_num += 1
        return x


class VGG(nn.Module):

    def __init__(self, block, args, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()

        print(f"\033[91mCreate VGG, block: {block}, cfg: {cfg}, num_classes: {num_classes} \033[0m")

        self.args = args 
        self.layer0 = self._make_layer0(VGGBlock, cfg[0], batch_norm, 3)
        self.layer1 = self._make_layers(block, cfg[1], batch_norm, cfg[0][-1])
        self.layer2 = self._make_layers(block, cfg[2], batch_norm, cfg[1][-1])
        self.layer3 = self._make_layers(block, cfg[3], batch_norm, cfg[2][-1])
        self.layer4 = self._make_layers(block, cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
    

    def _make_layer0(self, block, cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            layers.append(block(in_channels, v, batch_norm, self.args))
            in_channels = v
        return nn.Sequential(*layers)

    def _make_layers(self, block, cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            layers.append(block(in_channels, v, batch_norm, self.args))
            in_channels = v
        return MySequential(*layers)


    def set_replacing_rate(self, replacing_rate):
        self.args.replacing_rate = replacing_rate


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.layer0)
        feat_m.append(self.pool0)
        feat_m.append(self.layer1)
        feat_m.append(self.pool1)
        feat_m.append(self.layer2)
        feat_m.append(self.pool2)
        feat_m.append(self.layer3)
        feat_m.append(self.pool3)
        feat_m.append(self.layer4)
        feat_m.append(self.pool4)
        return feat_m
    

    def forward(self, x, save_dict=None, lambda_dict=None, is_feat=False, preact=False, flatGroupOut=False):
        h = x.shape[2]
        x = self.layer0(x)
        f0 = x # for distillation
        x = self.pool0(x)
        
        if save_dict:
            save_dict["layer_num"] = 1
        x = self.layer1(x, save_dict, lambda_dict)
        f1_pre = x # for distillation
        f1 = x # for distillation
        x = self.pool1(x)

        if save_dict:
            save_dict["layer_num"] = 2
        x = self.layer2(x, save_dict, lambda_dict)
        f2_pre = x # for distillation
        f2 = x # for distillation
        x = self.pool2(x)


        if save_dict:
            save_dict["layer_num"] = 3
        x = self.layer3(x, save_dict, lambda_dict)
        f3_pre = x # for distillation
        f3 = x # for distillation
        if h == 64:
            x = self.pool3(x)

        if save_dict:
            save_dict["layer_num"] = 4
        x = self.layer4(x, save_dict, lambda_dict)
        f4_pre = x # for distillation
        f4 = x # for distillation
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        f5 = x # for distillation

        x = self.classifier(x)

        # for crdst
        block_out0 = [block.out for block in self.layer0]
        block_out1 = [block.out for block in self.layer1]
        block_out2 = [block.out for block in self.layer2]
        block_out3 = [block.out for block in self.layer3]
        block_out4 = [block.out for block in self.layer4]

        # for crdst
        if flatGroupOut:
            f0_temp = self.pool4(f0)
            f0 = f0_temp.view(f0_temp.size(0), -1)
            
            f1_temp = self.pool4(f1)
            f1 = f1_temp.view(f1_temp.size(0), -1)
            
            f2_temp = self.pool4(f2)
            f2 = f2_temp.view(f2_temp.size(0), -1)
            
            f3_temp = self.pool4(f3)
            f3 = f3_temp.view(f3_temp.size(0), -1)
            
            f4_temp = self.pool4(f4)
            f4 = f4_temp.view(f4_temp.size(0), -1)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], [block_out1, block_out2, block_out3, block_out4], x 
            else:
                return [f0, f1, f2, f3, f4, f5], [block_out1, block_out2, block_out3, block_out4], x
        else:
            return x

cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]], # vgg-11
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]], # vgg-13
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]], # vgg-16
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]], # vgg-19
    'S': [[64], [128], [256], [512], [512]], # vgg-8
}



# =======> vgg8_bn
def vgg8_bn_fp(args):
    """VGG 8-layer model (configuration "S")"""
    model = VGG(VGGBlock, args, cfg['S'], batch_norm=True, num_classes=args.num_classes)
    return model


def vgg8_bn_quant(args):
    """VGG 8-layer model (configuration "S")"""
    model = VGG(QVGGBlock, args, cfg['S'], batch_norm=True, num_classes=args.num_classes)
    return model


# =======> vgg11_bn
def vgg11_bn_fp(args):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(VGGBlock, args, cfg['A'], batch_norm=True, num_classes=args.num_classes)
    return model


def vgg11_bn_quant(args):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(QVGGBlock, args, cfg['A'], batch_norm=True, num_classes=args.num_classes)
    return model


# =======> vgg13_bn
def vgg13_bn_fp(args):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(VGGBlock, args, cfg['B'], batch_norm=True, num_classes=args.num_classes)
    return model


def vgg13_bn_quant(args):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(QVGGBlock, args, cfg['B'], batch_norm=True, num_classes=args.num_classes)
    return model


# =======> vgg16_bn
def vgg16_bn_fp(args):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(VGGBlock, args, cfg['D'], batch_norm=True, num_classes=args.num_classes)
    return model


def vgg16_bn_quant(args):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(QVGGBlock, args, cfg['D'], batch_norm=True, num_classes=args.num_classes)
    return model


# =======> vgg19_bn
def vgg19_bn_fp(args):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(VGGBlock, args, cfg['E'], batch_norm=True, num_classes=args.num_classes)
    return model


def vgg19_bn_quant(args):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(QVGGBlock, args, cfg['E'], batch_norm=True, num_classes=args.num_classes)
    return model



if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    args = parser.parse_args()

    x = torch.randn(2, 3, 32, 32)
    net = vgg16_bn_fp(args)
    logit = net(x)

    # for name, p in net.named_parameters():
    #     print(f"{name:50} | {str(p.shape):50} | {p.requires_grad}")

    num_parameters = sum(p.numel() for p in net.parameters())
    # num_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('\nTotal number of parameters:', num_parameters)