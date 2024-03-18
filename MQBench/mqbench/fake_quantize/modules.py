import torch
import sys



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
        return grad_inputs, None, None, None, None, None
        

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
        return grad_inputs, None, None, None, None, None