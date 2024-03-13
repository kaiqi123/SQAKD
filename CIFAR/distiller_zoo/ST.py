from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import random

class SimilarityTransfer(nn.Module):
    def __init__(self, method, model_s):
        super(SimilarityTransfer, self).__init__()
        self.model_s = model_s
        self.method = method
        self.avgpool = self.determine_avgpool(model_s)
    
    def determine_avgpool(self, model_s):
        if "vgg" in model_s:
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif "resnet" in model_s:
            avgpool = nn.AvgPool2d(8)
        else:
            raise NotImplementedError(model_s)
        return avgpool

    def rmse_loss(self, s, t):
        return torch.sqrt(torch.mean((s-t)**2)) 


    def normalize_rmse(self, s, t):
        s = F.normalize(s, p=2, dim=0)
        t = F.normalize(t, p=2, dim=0)
        return self.rmse_loss(s, t)

    
    def forward(self, block_out_s, block_out_t):
        layer_pair_list = []
        for s, t in zip(block_out_s, block_out_t):
            s_norm, t_norm_mapped = self.st_loss_each_group(s, t)
            layer_pair_list.append((s_norm, t_norm_mapped))
        return layer_pair_list


    def st_loss_each_group(self, s_blocks, t_blocks):
        s_norm = F.normalize(s_blocks[-1], p=2, dim=0) 
        assert s_blocks[-1].shape == s_norm.shape
        t_norm_dict = {}
        simi_dict = {}
        for i in range(len(t_blocks)):
            t_norm = F.normalize(t_blocks[i], p=2, dim=0)
            t_norm_dict[i] = t_norm
            if s_norm.shape != t_norm.shape:
                s_norm = self.zero_pad_on_filter(s_norm, s_norm.shape[1], t_norm.shape[1])
            assert t_norm.shape == s_norm.shape
            simi_dict[i] = self.cosineSimilarity(s_norm, t_norm)

        if self.method == "Last":
            max_key = len(t_blocks)-1
        elif self.method == "Smallest":
            max_key = sorted(simi_dict.items(), key=lambda x: x[1], reverse=True).pop()[0] # reverse=False: largest; reverse=True: smallest
        elif self.method == "Largest":
            max_key = sorted(simi_dict.items(), key=lambda x: x[1], reverse=False).pop()[0] # reverse=False: largest; reverse=True: smallest
        elif self.method == "First":
            max_key = 0
        elif self.method == "Random":
            max_key = random.randint(0, len(t_blocks)-1)
        else:
            raise EOFError("method is not correct!")
        t_norm_mapped = t_norm_dict[max_key]

        s_norm_temp = self.avgpool(s_norm)
        s_norm = s_norm_temp.view(s_norm_temp.size(0), -1)
        t_norm_mapped_temp = self.avgpool(t_norm_mapped)
        t_norm_mapped = t_norm_mapped_temp.view(t_norm_mapped_temp.size(0), -1)
        
        return s_norm, t_norm_mapped

    def cosineSimilarity(self, x1, x2):
        x1_sqrt = torch.sqrt(torch.sum(x1 ** 2))
        x2_sqrt = torch.sqrt(torch.sum(x2 ** 2))
        return torch.div(torch.sum(x1 * x2), max(x1_sqrt * x2_sqrt, 1e-8))

    def zero_pad_on_filter(self, inputs, in_filter, out_filter):
        paddings = (0, 0, 0, 0, (out_filter - in_filter) // 2, (out_filter - in_filter) // 2, 0, 0)
        outputs = F.pad(inputs, paddings)
        return outputs