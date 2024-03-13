import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import sys
import json
import os

__all__ = ['QConv']


class EWGS_discretizer(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, x_in, num_levels, scaling_factor, save_dict=None, u=None, lambda_dict=None):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        diff = x_in - x_out
        ctx._scaling_factor = scaling_factor
        
        # new add for record u
        ctx._save_dict = save_dict
        if save_dict != None:
            layer_num = save_dict["layer_num"]
            block_num = save_dict["block_num"]
            conv_num = save_dict["conv_num"]
            w_activ_type = save_dict["type"]
            save_name = f"layer{layer_num}.block{block_num}.conv{conv_num}/{w_activ_type}_"
            ctx._save_name = save_name
        
        ctx.save_for_backward(diff)
        return x_out

    @staticmethod
    def backward(ctx, g):
        # EWGS
        diff = ctx.saved_tensors[0]
        save_dict = ctx._save_dict
        delta = ctx._scaling_factor
        u = g * delta * torch.sign(g)
        g_out = g + u * diff
        
        # for recording u_mean and u_std in each layer
        if save_dict != None:
            save_name = ctx._save_name
            iteration = save_dict["iteration"]
            writer = save_dict["writer"]
            u_mean = torch.mean(u).item()
            u_std = torch.std(u).item()
            writer.add_scalar(f"{save_name}u_mean", u_mean, iteration)
            writer.add_scalar(f"{save_name}u_std", u_std, iteration)

        return g_out, None, None, None, None, None


class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None


def quantize_k(r_i, k):
    scale = (2**k - 1)
    r_o = torch.round( scale * r_i ) / scale
    return r_o


# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        self.layer_name = None # for save layer_name, can be removed

        if self.quan_weight:
            self.weight_levels = args.weight_levels
            self.uW = nn.Parameter(data = torch.tensor(0).float())
            self.lW = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())


        if self.quan_act:
            self.act_levels = args.act_levels
            self.uA = nn.Parameter(data = torch.tensor(0).float())
            self.lA = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorA', torch.tensor(args.bkwd_scaling_factorA).float())

            self.uA_t = nn.Parameter(data = torch.tensor(0).float())
            self.lA_t = nn.Parameter(data = torch.tensor(0).float())

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.Parameter(data = torch.tensor(1).float())
        
        self.hook_Qvalues = False
        self.buff_weight = None
        self.buff_act = None



    def weight_quantization(self, weight, save_dict=None, lambda_dict=None, p=1):

        weight = (weight - self.lW) / (self.uW - self.lW)
        weight = weight.clamp(min=0, max=1) # [0, 1]

        if not self.baseline:
            if save_dict:
                save_dict["type"] = "weight"
            weight = self.EWGS_discretizer(weight, self.weight_levels, self.bkwd_scaling_factorW, save_dict, None, None)
        else:
            weight = self.STE_discretizer(weight, self.weight_levels)
            
        if self.hook_Qvalues:
            self.buff_weight = weight
            self.buff_weight.retain_grad()
        
        weight = (weight - 0.5) * 2


        return weight

    def act_quantization(self, x, save_dict=None, lambda_dict=None, p=1):
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1) # [0, 1]
        
        if not self.baseline:
            if save_dict:
                save_dict["type"] = "activ"
            x = self.EWGS_discretizer(x, self.act_levels, self.bkwd_scaling_factorA, save_dict, None, None)
        else:
            x = self.STE_discretizer(x, self.act_levels)

        if self.hook_Qvalues:
            self.buff_act = x
            self.buff_act.retain_grad()

        return x


    def initialize(self, x):

        Qweight = self.weight
        Qact = x
        
        if self.quan_weight:
            self.uW.data.fill_(self.weight.std()*3.0)
            self.lW.data.fill_(-self.weight.std()*3.0)
            Qweight = self.weight_quantization(self.weight)

        if self.quan_act:
            self.uA.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.lA.data.fill_(x.min())
            Qact = self.act_quantization(x)
            
            self.uA_t.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.lA_t.data.fill_(x.min())

        Qout = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        out = F.conv2d(x, self.weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        self.output_scale.data.fill_(out.abs().mean() / Qout.abs().mean())


    def forward(self, x, save_dict=None, lambda_dict=None):
        # for p, can be removed
        p = 1

        # for saving layer_name, can be removed
        if save_dict:
            layer_num = save_dict["layer_num"]
            block_num = save_dict["block_num"]
            conv_num = save_dict["conv_num"]
            self.layer_name = f"layer{layer_num}.block{block_num}.conv{conv_num}"

        if self.init == 1:
            self.initialize(x)
        
        Qweight = self.weight
        if self.quan_weight:
            Qweight = self.weight_quantization(Qweight, save_dict, lambda_dict, p)
    

        Qact = x
        if self.quan_act:
            Qact = self.act_quantization(Qact, save_dict, lambda_dict, p)


        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups) * torch.abs(self.output_scale)

        return output