import torch
from torch import nn
import math
import sys

class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        """
        inputSize: opt.feat_dim, 128
        outputSize: n_data, number of training data, cifar100: 50000
        """
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        """
        Refer to CRDLoss forward()
        Args:
            v1---f_s: the feature of student network, size [batch_size, s_dim]
            v2---f_t: the feature of teacher network, size [batch_size, t_dim]
            y ---idx: the indices of these positive samples in the dataset, size [batch_size]
            idx---contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        """
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()

        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0) 
        inputSize = self.memory_v1.size(1) 

        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach() # shape: [65540, 128]
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize) # shape: [4, 16385, 128]
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1)) # shape: [4, 16385, 1]
        out_v2 = torch.exp(torch.div(out_v2, T)) # shape: [4, 16385, 1]
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T)) # shape: [4, 16385, 1]
        
        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1)) # 72995.1
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2)) # 73917.7

        # compute out_v1, out_v2
        # contiguous(): returns itself if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.
        out_v1 = torch.div(out_v1, Z_v1).contiguous() 
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            # memory_v1: [5000, 128]
            # l_pos = select positive index from memory_v1
            # updated_v1 = Norm[l_pos * momentum + v1 * (1-momentum)]
            # put updated_v1 back to memory_v1
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1)) # [4, 128], y.view(-1): tensor([ 2120, 11102, 33744,  1827], device='cuda:0')
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum)) # v1: [4,128]
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm) # [4,128]
            self.memory_v1.index_copy_(0, y, updated_v1) # y: tensor([ 2120, 11102, 33744,  1827], device='cuda:0')
            
            # ab_pos = select positive index from memory_v2
            # updated_v2 = Norm[ab_pos * momentum + v2 * (1-momentum)]
            # put updated_v2 back to memory_v2
            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    Refer to: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    Refer to: https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj