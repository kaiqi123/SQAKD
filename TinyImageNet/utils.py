import torch
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torchvision.transforms as transforms

# new
import numpy as np
import logging
import os
_logger = None
_logger_fh = None
_logger_names = []

# new
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def printRed(skk): print("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m{}\033[00m" .format(skk))



def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

# following adjust_learning_rate() in CRD code
def adjust_learning_rate_crd(epoch, opt, optimizer, logger=None):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr    
    
    # # new add, log lr
    # for i, param_group in enumerate(optimizer.param_groups):
    #     logger.log_value(f'{i}_learning_rate_epoch', param_group['lr'], epoch)
    #     # print(f"{i}_learning_rate_epoch, {param_group['lr']}, {epoch}")


class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(F.log_softmax(input, 1) * (one_hot.detach())) / input.size(0)
        return loss

SyncBatchNorm2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm

def replace_bn_to_syncbn(model, custombn=SyncBatchNorm2d):
    if type(model) in [torch.nn.BatchNorm2d]:
        return _replace_bn(model, custombn)

    elif type(model) in [torch.nn.intrinsic.qat.ConvBn2d, torch.nn.intrinsic.qat.ConvBnReLU2d]:
        model.bn = _replace_bn(model.bn, custombn)
        return model

    elif type(model) in [torch.nn.intrinsic.BNReLU2d]:
        model[0] = _replace_bn(model[0], custombn)
        return model

    else:
        for name, module in model.named_children():
            setattr(model, name, replace_bn_to_syncbn(module))
        return model


def _replace_bn(bn, custombn):
    syncbn = custombn(bn.num_features, bn.eps, bn.momentum, bn.affine)
    # syncbn = custombn(bn.num_features, bn.eps, bn.momentum, bn.affine)
    if bn.affine:
        syncbn.weight = bn.weight
        syncbn.bias = bn.bias
    syncbn.running_mean = bn.running_mean
    syncbn.running_var = bn.running_var
    return syncbn


def makedir(path, local_rank):
    if local_rank == 0 and not os.path.exists(path):
        os.makedirs(path)

def create_logger(log_file, level=logging.INFO):
    global _logger, _logger_fh
    if _logger is None:
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.setLevel(level)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _logger_fh = fh
    else:
        _logger.removeHandler(_logger_fh)
        _logger.setLevel(level)

    return _logger

def get_logger(name, level=logging.INFO):
    global _logger_names
    logger = logging.getLogger(name)
    # logger.parent = None
    if name in _logger_names:
        return logger

    _logger_names.append(name)
    # if link.get_rank() > 0:
    #     logger.addFilter(RankFilter())

    return logger