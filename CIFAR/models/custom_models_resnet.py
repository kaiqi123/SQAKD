import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys

from .custom_modules import QConv
from .blocks_resnet import BasicBlock, QBasicBlock



__all__ = ['resnet20_quant', 'resnet20_fp', 'resnet32_quant', 'resnet32_fp', 'resnet18_fp', 'resnet18_quant']



def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, QConv):
        nn.init.kaiming_normal_(m.weight)



class MySequential(nn.Sequential):
    def forward(self, x, save_dict=None, lambda_dict=None):
        block_num = 0
        for module in self._modules.values():
            if save_dict:
                save_dict["block_num"] = block_num
            x = module(x, save_dict, lambda_dict)
            block_num += 1
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()

        self.args = args
        num_classes = args.num_classes

        print(f"\033[91mCreate ResNet, block: {block}, num_blocks: {num_blocks}, num_classes: {num_classes} \033[0m")

        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.args, stride))
            self.in_planes = planes * block.expansion

        return MySequential(*layers)
    

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(nn.ReLU(inplace=True)) 
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def set_replacing_rate(self, replacing_rate):
        self.args.replacing_rate = replacing_rate

    def forward(self, x, save_dict=None, lambda_dict=None, is_feat=False, preact=False, flatGroupOut=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out

        if save_dict:
            save_dict["layer_num"] = 1
        out = self.layer1(out, save_dict, lambda_dict)
        f1 = out

        if save_dict:
            save_dict["layer_num"] = 2
        out = self.layer2(out, save_dict, lambda_dict)
        f2 = out

        if save_dict:
            save_dict["layer_num"] = 3
        out = self.layer3(out, save_dict, lambda_dict)
        f3 = out

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        f4 = out

        out = self.bn2(out)
        out = self.linear(out)

        block_out1 = [block.out for block in self.layer1]
        block_out2 = [block.out for block in self.layer2]
        block_out3 = [block.out for block in self.layer3]

        if flatGroupOut:
            f0_temp = nn.AvgPool2d(8)(f0)
            f0 = f0_temp.view(f0_temp.size(0), -1)
            f1_temp = nn.AvgPool2d(8)(f1)
            f1 = f1_temp.view(f1_temp.size(0), -1)
            f2_temp = nn.AvgPool2d(8)(f2)
            f2 = f2_temp.view(f2_temp.size(0), -1)
            f3_temp = nn.AvgPool2d(8)(f3)
            f3 = f3_temp.view(f3_temp.size(0), -1)

        if is_feat:
            if preact:
                raise NotImplementedError(f"{preact} is not implemented")
            else:
                return [f0, f1, f2, f3, f4], [block_out1, block_out2, block_out3], out
        else:
            return out


# resnet20
def resnet20_fp(args):
    return ResNet(BasicBlock, [3, 3, 3], args)


def resnet20_quant(args):
    return ResNet(QBasicBlock, [3, 3, 3], args)


# resnet32
# n = (depth - 2) // 6
def resnet32_fp(args):
    return ResNet(BasicBlock, [5, 5, 5], args)

def resnet32_quant(args):
    return ResNet(QBasicBlock, [5, 5, 5], args)



# resnet18
def resnet18_fp(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args)

def resnet18_quant(args):
    return ResNet(QBasicBlock, [2, 2, 2, 2], args)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    '''
    resnet20_fp
    Total number of params 269850
    Total layers 20
    '''
    import argparse
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Test Model full position")
    parser.add_argument('--quan_method', type=str, default='EWGS', choices=['PACT', 'EWGS', 'LSQ'], help='training with different quantization methods')
    parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
    parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
    parser.add_argument('--weight_levels', type=int, default=2, help='number of weight quantization levels')
    parser.add_argument('--act_levels', type=int, default=2, help='number of activation quantization levels')
    parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
    parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
    parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
    parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scsaling factor using Hessian trace')
    parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')
    args = parser.parse_args()
    args.num_classes = 10

    # test fp
    # net = resnet20_fp(args) 
    # test(net)

    # test EWGS
    net = resnet20_quant(args)
    test(net)
    
    for m in net.modules():
        print(m)
    # for name, p in net.named_parameters():
    #     print(f"{name:50}, {p.shape}")