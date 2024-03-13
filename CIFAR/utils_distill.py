import torch.nn as nn
import torch

import sys

from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from distiller_zoo import SimilarityTransfer
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import FactorTransfer, FSP, NSTLoss
from crd.criterion import CRDLoss

from utils import printRed
 
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn



# for distillation, only use when args.distill is not None
def define_distill_module_and_loss(model_s, model_t, model_params, args, n_data, train_loader):
    printRed("Define distillation modules and loss terms")
    flatGroupOut = True if args.distill == 'crdst' else False
    data = torch.randn(2, 3, 32, 32).cuda() 
    model_t.eval()
    model_s.eval()
    feat_t, block_out_t, _ = model_t(data, is_feat=True, flatGroupOut=flatGroupOut)
    feat_s, block_out_s, _ = model_s(data, is_feat=True, flatGroupOut=flatGroupOut)

    module_list = nn.ModuleList([])
    module_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    if args.distill == 'kd':
        criterion_kd = DistillKL(args.kd_T)

    elif args.distill == 'crd':
        args.s_dim = feat_s[-1].shape[1]
        args.t_dim = feat_t[-1].shape[1]
        args.n_data = n_data # number of training data
        criterion_kd = CRDLoss(args)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)

        for name, param in criterion_kd.embed_s.named_parameters():
            model_params.append(param)
        for name, param in criterion_kd.embed_t.named_parameters():
            model_params.append(param)

    elif args.distill == 'crdst':
        similarity_transfer = SimilarityTransfer(args.st_method, args.arch)
        criterion_kd = nn.ModuleList([])
        criterion_kd.append(similarity_transfer)
        for i in range(len(feat_s)):
            if i < len(feat_s)-1:
                args.s_dim = feat_t[i].shape[1]
            else:
                args.s_dim = feat_s[i].shape[1]
            args.t_dim = feat_t[i].shape[1]
            args.n_data = n_data
            criterion_kd_single = CRDLoss(args)
            module_list.append(criterion_kd_single.embed_s)
            module_list.append(criterion_kd_single.embed_t)
            criterion_kd.append(criterion_kd_single)
            for name, param in criterion_kd_single.embed_s.named_parameters():
                model_params.append(param)
            for name, param in criterion_kd_single.embed_t.named_parameters():
                model_params.append(param)
        
        
    elif args.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape)
        module_list.append(regress_s)
        for name, param in regress_s.named_parameters():
            print(name, param.shape)
            model_params.append(param)

    elif args.distill == 'attention':
        criterion_kd = Attention()

    elif args.distill == 'nst':
        criterion_kd = NSTLoss()

    elif args.distill == 'similarity':
        criterion_kd = Similarity()

    elif args.distill == 'rkd':
        criterion_kd = RKDLoss()

    elif args.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], args.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], args.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        for name, param in embed_s.named_parameters():
            model_params.append(param)
        for name, param in embed_t.named_parameters():
            model_params.append(param)

    elif args.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        for name, param in criterion_kd.named_parameters():
            model_params.append(param)

    elif args.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, args)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        for name, param in translator.named_parameters():
            model_params.append(param)

    elif args.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, args)
        # classification training
        pass
    else:
        raise NotImplementedError(args.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    return module_list, model_params, criterion_list



def get_loss_kd(args, feat_s, feat_t, criterion_kd, module_list, index, contrast_idx):

    if args.distill == 'kd':
        loss_kd = 0

    elif args.distill == 'crd':
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

    elif args.distill == 'hint':
        f_s = module_list[1](feat_s[args.hint_layer])
        f_t = feat_t[args.hint_layer]
        loss_kd = criterion_kd(f_s, f_t)
    elif args.distill == 'attention':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'nst':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'similarity':
        g_s = [feat_s[-2]]
        g_t = [feat_t[-2]]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'rkd':
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t)
    elif args.distill == 'pkt':
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t)
    elif args.distill == 'kdsvd':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    elif args.distill == 'correlation':
        f_s = module_list[1](feat_s[-1])
        f_t = module_list[2](feat_t[-1])
        loss_kd = criterion_kd(f_s, f_t)
    elif args.distill == 'vid':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
        loss_kd = sum(loss_group)
    elif args.distill == 'abound':
        # can also add loss to this stage
        loss_kd = 0
    elif args.distill == 'fsp':
        # can also add loss to this stage
        loss_kd = 0
    elif args.distill == 'factor':
        factor_s = module_list[1](feat_s[-2])
        factor_t = module_list[2](feat_t[-2], is_factor=True)
        loss_kd = criterion_kd(factor_s, factor_t)
    else:
        raise NotImplementedError(args.distill)

    return loss_kd


def get_loss_crdst(args, feat_s, feat_t, criterion_kd, index, contrast_idx, block_out_s, block_out_t):
    assert args.distill == 'crdst'
    layer_pair_list = criterion_kd[0](block_out_s, block_out_t)

    loss_kd_crdSt_list = []

    f0_s = feat_s[0]
    f0_t = feat_t[0]
    loss_kd_crdSt_list.append(criterion_kd[1](f0_s, f0_t, index, contrast_idx))
    
    for i in range(2, len(layer_pair_list)+2): 
        f_s, f_t = layer_pair_list[i-2]
        loss_kd_crdSt_list.append(criterion_kd[i](f_s, f_t, index, contrast_idx))

    f_s = feat_s[-1]
    f_t = feat_t[-1]
    loss_kd_crd = criterion_kd[-1](f_s, f_t, index, contrast_idx)
    
    loss_kd_crdSt = sum(loss_kd_crdSt_list)
    return loss_kd_crd, loss_kd_crdSt



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init(model_s, model_t, init_modules, criterion, train_loader, args):
    printRed("Init modules for abound, factor, fsp")
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True


    model_name = args.arch.split("_")[0]
    if model_name in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            args.distill == 'factor':
        lr = 0.01
    else:
        lr = args.lr_m
    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, args.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if args.distill == 'crd' or args.distill == 'crdst':
                input, target, index, contrast_idx = data
            else:
                input, target = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                # index = index.cuda()
                if args.distill == 'crd' or args.distill == 'crdst':
                    contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            preact = (args.distill == 'abound')
            feat_s, _, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if args.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif args.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif args.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(args.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, args.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()