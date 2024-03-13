import torch
# import torch.nn.functional as F
# import math
# import numpy as np

import sys
# import os
# import json

# from .utils import grad_scale


class FakeAffineTensorQuantFunction_STE(torch.autograd.Function):
    """Fake version of affine quantization
    Refer to: 
    https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/_modules/pytorch_quantization/tensor_quant.html#ScaledQuantDescriptor
    https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    # def forward(ctx, inputs, min_range, max_range, num_bits=8):
    def forward(ctx, input, scale, zero_point, quant_min, quant_max, grad_factor=None):
        # ctx.save_for_backward(input, quant_min, quant_max)
        ctx.save_for_backward(input)
        ctx._quant_min = quant_min
        ctx._quant_max = quant_max
        # if grad_factor:
        #     scale = grad_scale(scale, grad_factor) 
        #     zero_point = grad_scale(zero_point, grad_factor) 
        quantized = torch.round(input / scale) - zero_point
        quantized = torch.clamp(quantized, quant_min, quant_max)
        output = (quantized + zero_point) * scale
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        # inputs, min_range, max_range = ctx.saved_tensors
        input = ctx.saved_tensors[0]
        quant_min = ctx._quant_min
        quant_max = ctx._quant_max
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where((input <= quant_max)*(input >= quant_min), grad_outputs, zero)
        # print("STE")
        return grad_inputs, None, None, None, None, None
        


class FakeAffineTensorQuantFunction_EWGS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, quant_min, quant_max, scaling_factor=0.001, grad_factor=None):
        ctx._quant_min = quant_min
        ctx._quant_max = quant_max
        ctx._scaling_factor = scaling_factor
        # if grad_factor:
        #     scale = grad_scale(scale, grad_factor) 
        #     zero_point = grad_scale(zero_point, grad_factor)
        quantized = torch.round(input / scale) - zero_point
        quantized = torch.clamp(quantized, quant_min, quant_max)
        output = (quantized + zero_point) * scale
        diff = input - output
        ctx.save_for_backward(input, diff)
        return output

    @staticmethod
    def backward(ctx, grad):
        input, diff = ctx.saved_tensors
        quant_min = ctx._quant_min
        quant_max = ctx._quant_max
        delta = ctx._scaling_factor
        zero = grad.new_zeros(1)
        grad_inputs = torch.where((input <= quant_max)*(input >= quant_min), grad, zero)
        
        u = grad_inputs * delta * torch.sign(grad_inputs)
        grad_outputs = grad_inputs + u * diff
        
        # for saving u_mean
        # u_mean = torch.mean(u).item()
        # print("EWGS", u_mean)
        return grad_outputs, None, None, None, None, None, None


class FakeAffineTensorQuantFunction_Uscheduler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, quant_min, quant_max, u, grad_factor=None):
        ctx._quant_min = quant_min
        ctx._quant_max = quant_max
        ctx._u = u
        # if grad_factor:
        #     scale = grad_scale(scale, grad_factor) 
        #     zero_point = grad_scale(zero_point, grad_factor)
        quantized = torch.round(input / scale) - zero_point
        quantized = torch.clamp(quantized, quant_min, quant_max)
        output = (quantized + zero_point) * scale
        diff = input - output
        ctx.save_for_backward(input, diff)
        return output

    @staticmethod
    def backward(ctx, grad):
        input, diff = ctx.saved_tensors
        quant_min = ctx._quant_min
        quant_max = ctx._quant_max
        u = ctx._u
        zero = grad.new_zeros(1)
        grad_inputs = torch.where((input <= quant_max)*(input >= quant_min), grad, zero)
        
        grad_outputs = grad_inputs + u * diff
        # print("Uscheduler", u)
        return grad_outputs, None, None, None, None, None, None


# =========================================================================================================
# def _fake_quantize_learnable_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
#     zero_point = (zero_point.round() - zero_point).detach() + zero_point
#     new_shape = [1] * len(x.shape)
#     new_shape[ch_axis] = x.shape[ch_axis]
#     scale = grad_scale(scale, grad_factor).reshape(new_shape)
#     zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
#     x = x / scale + zero_point
#     x = (x.round() - x).detach() + x
#     x = torch.clamp(x, quant_min, quant_max)
#     return (x - zero_point) * scale
class fake_quantize_learnable_per_tensor_affine_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, quant_min, quant_max, grad_factor=None):
        # ctx.save_for_backward(input, quant_min, quant_max)
        zero_point = (zero_point.round() - zero_point).detach() + zero_point # new
        ctx.save_for_backward(input)
        ctx._quant_min = quant_min
        ctx._quant_max = quant_max
        if grad_factor:
            scale = grad_scale(scale, grad_factor) 
            zero_point = grad_scale(zero_point, grad_factor) 
        quantized = torch.round(input / scale) - zero_point
        quantized = (quantized.round() - quantized).detach() + quantized # new
        quantized = torch.clamp(quantized, quant_min, quant_max)
        output = (quantized + zero_point) * scale
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        # inputs, min_range, max_range = ctx.saved_tensors
        input = ctx.saved_tensors[0]
        quant_min = ctx._quant_min
        quant_max = ctx._quant_max
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where((input <= quant_max)*(input >= quant_min), grad_outputs, zero)
        # print("STE")
        return grad_inputs, None, None, None, None, None