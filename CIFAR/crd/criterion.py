import torch
from torch import nn
from .memory import ContrastMemory
import sys

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        if opt.head == "linear":
            self.embed_s = Embed(opt.s_dim, opt.feat_dim)
            self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        elif opt.head == "mlp":
            self.embed_s = Embed_mlp(opt.s_dim, opt.feat_dim)
            self.embed_t = Embed_mlp(opt.t_dim, opt.feat_dim)
        elif opt.head == "pad":
            self.embed_s = Embed_pad(opt.s_dim, opt.feat_dim)
            self.embed_t = Embed_pad(opt.t_dim, opt.feat_dim)
        else:
            raise NotImplementedError(f'head not supported: {opt.head}') 
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s) # [bs, 128]
        f_t = self.embed_t(f_t) # [bs, 128]
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx) # [4, 16385, 1]
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1 # 16384

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        # log_D1 = log[P_pos / (P_pos + nce_k/n_data)]
        P_pos = x.select(1, 0) # [4, 1], select positive
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()  # [4, 1]
        

        # loss for K negative pair
        # log_D0 = log[ (nce_k/n_data) / (P_neg + nce_k/n_data) ]
        P_neg = x.narrow(1, 1, m) # [4, 16384, 1]
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_() # [4, 16384, 1]
        
        
        # log_D1.sum(0): tensor([-39.1143], device='cuda:0', grad_fn=<SumBackward1>)
        # log_D0.view(-1, 1).shape: [65536, 1]
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Embed_pad(nn.Module):
    """Embed_padding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed_pad, self).__init__()
        self.l2norm = Normalize(2)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def zero_pad(self, inputs, dim_in, dim_out):
        paddings = ((dim_out - dim_in) // 2, (dim_out - dim_in) // 2)
        outputs = torch.nn.functional.pad(inputs, paddings)
        return outputs

    def forward(self, x):
        x = x.view(x.shape[0], -1) # [bs, dim_in]; linear: [dim_in, dim_out]
        x = self.zero_pad(x, self.dim_in, self.dim_out) # [bs, dim_out]
        x = self.l2norm(x)
        return x


class Embed_mlp(nn.Module):
    """Embed_mlp module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed_mlp, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_in)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
