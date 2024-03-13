import argparse
import os
import shutil
import time
import math
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization, disable_all

from get_config import get_extra_config
from utils import LabelSmoothCELoss
from lr_scheduler import Cosine

import numpy as np
import json
import gc

# import models.resnet_imagenet as resnet_imagenet_models
# import torchvision.models as models # official models
from models import model_dict # our customized models

from data_loaders.cifar_data_loader import cifar_data_loader
from data_loaders.imagenet_data_loader import imagenet_data_loader

# try:
#     from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
#     from nvidia.dali.pipeline import pipeline_def
#     import nvidia.dali.types as types
#     import nvidia.dali.fn as fn
# except ImportError:
#     raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

# new
from tensorboardX import SummaryWriter
import pprint
from utils import printRed, makedir, create_logger, get_logger
import timm
from GPUtil import showUtilization as gpu_usage

# new add for kd
from distiller_zoo import DistillKL

# new add for inference time
from utils_measure_inference import measure_inference



def parse():
    # model_names = sorted(name for name in models.__dict__
    #                  if name.islower() and not name.startswith("__")
    #                  and callable(models.__dict__[name])) # official models
    model_names = model_dict.keys() # our customized models
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='0.04,0.01,0.004,0.001,Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')


    parser.add_argument('--dali_cpu', action='store_true', help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--prof', default=-1, type=int, help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str, default=None)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true', help='Launch test mode with preset arguments')
    parser.add_argument('--not-quant', action='store_true')

    # new
    parser.add_argument('--save_root_path', type=str)
    parser.add_argument('--backward_method', type=str, choices=["org","STE","EWGS","Uscheduler","Uscheduler_Pscheduler"])

    # new for kd
    parser.add_argument('--distill', default=False, type=bool)
    parser.add_argument('--teacher_path', type=str)
    parser.add_argument('--teacher_arch', type=str)
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--gamma', type=float, default=None, help='weight for classification')
    parser.add_argument('--alpha', type=float, default=None, help='weight balance for KD')
    # parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses or crd loss')

    # new add for training, following EWGS
    parser.add_argument('--optimizer_type', type=str, default='SGD', choices=('SGD','Adam'), help='optimizer for model paramters')
    # parser.add_argument('--lr_scheduler_type', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
    # parser.add_argument('--lr_decay_schedule', type=str, help='learning rate decaying schedule (for step)')

    # new for intializing from a pretrained model, following EWGS
    parser.add_argument('--load_pretrain', action='store_true', help='load pretrained full-precision model')
    parser.add_argument('--pretrain_path', type=str, help='path for pretrained full-preicion model')

    args = parser.parse_args()
    return args

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def main():
    global best_prec1, best_prec5, args
    best_prec1 = 0
    best_prec5 = 0
    args = parse()
    args.quant = not args.not_quant

    
    # set environment
    root_path = os.path.join(os.getcwd(), args.save_root_path) # new
    save_path = os.path.join(root_path, 'checkpoints')
    event_path = os.path.join(root_path, 'events')
    makedir(save_path, args.local_rank)
    makedir(event_path, args.local_rank)
    # tb_logger
    if args.local_rank == 0:
        tb_logger = SummaryWriter(event_path)
    else:
        tb_logger = None
    # logger
    # create_logger(os.path.join(root_path, 'log.txt'))
    # logger = get_logger(__name__)
    # logger.info(f'config: {pprint.pformat(get_config())}')
    settings = vars(args)
    outfile = os.path.join(root_path, 'log.txt')
    with open(outfile, 'w') as convert_file:
        convert_file.write(f'Quantization config: \n{pprint.pformat(get_extra_config())}\n')
        convert_file.write(f'Training config: \n{pprint.pformat(settings)}')
        # convert_file.write('json_stats: ' + json.dumps(vars(args)) + '\n')
    f = open(outfile, 'a+')
    print('\n\n##################### time: {} ####################'.format(time.ctime()), file=f, flush=True)
    # sys.exit()

    # test mode, use default args for sanity test
    if args.test:
        args.opt_level = None
        args.epochs = 1
        args.start_epoch = 0
        args.arch = 'resnet50'
        args.batch_size = 64
        args.data = []
        args.sync_bn = False
        args.data.append('/data/imagenet/train-jpeg/')
        args.data.append('/data/imagenet/val-jpeg/')
        logger.info("Test mode - no DDP, no apex, RN50, 10 iterations")

    if not len(args.data):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    # make apex optional
    if args.opt_level is not None or args.distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    #print("world size = {}".format(int(os.environ['WORLD_SIZE'])))
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
        

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


    # Data loading code
    data_name = args.data[0].split("/")[-1]
    if data_name == "tiny-imagenet-200" or data_name == "imagenet_data":
        train_loader, val_loader = imagenet_data_loader(args)
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
        val_loader_len = int(val_loader._size / args.batch_size)
        data_loader_type = "dali"

    elif data_name == "cifar10" or data_name == "cifar100":
        train_loader, val_loader = cifar_data_loader(data_name, args)
        train_loader_len = len(train_loader)
        val_loader_len = len(val_loader)
        data_loader_type = "pytorch"

    else:
        raise NotImplementedError
    printRed(f"Data name is: {data_name}, data path is: {args.data}, length of train_loader: {train_loader_len}, length of val_loader_len: {val_loader_len}")


    # New add to set num_classes
    num_classes_dict = {"tiny-imagenet-200": 200, "imagenet_data": 1000, "cifar10": 10, "cifar100": 100}
    data_name = args.data[0].split("/")[-1]
    assert data_name in num_classes_dict.keys()
    num_classes = num_classes_dict[data_name]
    printRed(f"Dataset path is: {args.data}, num of classes is: {num_classes}")
    

    # create model
    # if args.pretrained:
    #     print("=> Using pre-trained model '{}'".format(args.arch))
    #     if "resnet" in args.arch:
    #         model = resnet_imagenet_models.__dict__[args.arch](pretrained=True)
    #     elif args.arch == "efficientnet_b0":
    #         model = timm.create_model('efficientnet_lite0', pretrained=True)
    #     elif args.arch == "mobilenet_v2":
    #         model = models.mobilenet_v2(pretrained=True, progress=True)
    #     else:
    #         raise Error(f"Do not support {args.arch}")
    # else:
    #     print("=> Not using pre-trained model, creating model '{}'".format(args.arch))
    #     if "resnet" in args.arch:
    #         model = resnet_imagenet_models.__dict__[args.arch](num_classes=num_classes) # Modified num_classes
    #     elif args.arch == "efficientnet_b0":
    #         model = timm.create_model('efficientnet_lite0', pretrained=False)
    #     elif args.arch == "mobilenet_v2":
    #         model = models.mobilenet_v2(pretrained=False)
    #     else:
    #         raise Error(f"Do not support {args.arch}")
    pretrained =  True if args.pretrained else False
    model = model_dict[args.arch](pretrained=pretrained, num_classes=num_classes) # our customized models, Modified num_classes
    # model = models.__dict__[args.arch](pretrained=pretrained, num_classes=num_classes) # official models
    # sys.exit()

    # initilized by pre-trained model
    if args.load_pretrain:
        trained_model = torch.load(args.pretrain_path)
        current_dict = model.state_dict()
        printRed(f"Initialized from the pretrained full precision weights: {args.pretrain_path}")
        print("Following modules are initialized from pretrained model: ")
        log_string = ''
        for key in trained_model['state_dict'].keys():
            if key in current_dict.keys():
                # print(key)
                log_string += '{}\t'.format(key)
                current_dict[key].copy_(trained_model['state_dict'][key])
        print(log_string+'\n')
        model.load_state_dict(current_dict)
        print(f"The best accuracy of the pretrained model is: {trained_model['best_prec1']}, from epoch: {trained_model['epoch']}")
    else:
        printRed("Not initialized from the pretrained full precision weights")
    

    # [pre1_s, pre5_s] = validate(val_loader, model.cuda(), nn.CrossEntropyLoss().cuda(), val_loader_len, data_loader_type)
    # printRed(f"Evaluate the initial model (student): Prec@1: {pre1_s}, Prec@5: {pre5_s}")


    # new added, create teacher
    if args.distill:
        # org
        # model_t = resnet_imagenet_models.__dict__[args.teacher_arch](num_classes=num_classes)
        if data_name == "imagenet_data":
            # for imagenet, load from official model
            printRed('Creating teacher model, loading from offical website')
            # model_t = model_dict[args.teacher_arch](pretrained=True, num_classes=num_classes) # our customized models, Modified num_classes
            model_t = models.__dict__[args.teacher_arch](pretrained=True, num_classes=num_classes) # official models
        else:
            # for other datasets, load from the model trained by ourselves
            printRed('Creating teacher model, loading from the model trained by ourselves')
            model_t = model_dict[args.teacher_arch](pretrained=False, num_classes=num_classes) # our customized models, Modified num_classes
            # model_t = models.__dict__[args.teacher_arch](pretrained=False, num_classes=num_classes) # official models
            if os.path.isfile(args.teacher_path):
                print("Loading checkpoint '{}'".format(args.teacher_path))
                checkpoint_t = torch.load(args.teacher_path, map_location = lambda storage, loc: storage.cuda(args.gpu))
                model_t.load_state_dict(checkpoint_t['state_dict'])
                print("Loaded, epoch: {}, acc: {})".format(checkpoint_t['epoch'], checkpoint_t['best_prec1']))
            else:
                raise("No checkpoint found at '{}'".format(args.teacher_path))
        
        [pre1_t, pre5_t] = validate(val_loader, model_t.cuda(), nn.CrossEntropyLoss().cuda(), val_loader_len, data_loader_type)
        printRed(f"Evaluate the teacher model: Prec@1: {pre1_t}, Prec@5: {pre5_t}")
        # sys.exit()

    if args.quant:
        # model = prepare_by_platform(model, BackendType.Tensorrt)
        extra_config = get_extra_config()
        model = prepare_by_platform(model, BackendType.Academic, extra_config, backward_method=args.backward_method)
        printRed(extra_config)
    else:
        printRed("Do not quantize, train the original model")


    if args.sync_bn:
        printRed("using apex synced BN")
        model = parallel.convert_syncbn_model(model)
        if args.distill:
            model_t = parallel.convert_syncbn_model(model_t)
    # replace_bn_to_syncbn(model)

    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
        if args.distill:
            model_t = model_t.cuda().to(memory_format=memory_format)
        
        # new add for inference
        # # cpu_name = "all" 
        # cpu_name = "0" # all for all cores, 0 for cpu0, command "htop" to see the utility of each cpu
        # os.system(f"taskset -p -c {cpu_name} {os.getpid()}")
        # device = torch.device("cpu")
        # model = model.to(device).to(memory_format=memory_format)
    else:
        model = model.cuda()
        if args.distill:
            model_t = model_t.cuda()
    # sys.exit()

    if args.distill:
        module_list = nn.ModuleList([])
        module_list.append(model)
        # trainable_list = nn.ModuleList([])
        # trainable_list.append(model)
    

    # Scale learning rate based on global batch size
    # args.lr = args.lr*float(args.batch_size*args.world_size)/256.

    # use for cosine, tiny-imagenet
    # should be 0.0001 here, otherwise lr will be a very large value at the beginning
    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 0.0001, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    printRed(f"Optimizer is: {args.optimizer_type}, weight_decay: {args.weight_decay}, momentum: {args.momentum}")


    # Define learning rate for consine, used for tiny-imagenet
    max_iter = train_loader_len * (args.epochs - args.start_epoch)
    lr_scheduler = Cosine(optimizer, max_iter = max_iter, min_lr = 0.0, base_lr = 0.0001, warmup_lr = args.lr, warmup_steps = 2500)
    print(f"learning rate: {args.lr}, epochs: {args.epochs}, max_iter: {max_iter}, train_loader_len: {train_loader_len}, bs: {args.batch_size}")


    # # use for step
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

    if args.distill:
        # append teacher after optimizer to avoid weight_decay
        module_list.append(model_t)


    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.

        # org
        # model = DDP(model, delay_allreduce=True)
        # new add
        if args.distill:
            for i, model in enumerate(module_list):
                module_list[i] = DDP(model, delay_allreduce=True)
        else:
            model = DDP(model, delay_allreduce=True)

    
    # define loss function (criterion)
    data_name = args.data[0].split("/")[-1]
    if data_name == "imagenet_data" or data_name == "tiny-imagenet-200":
        label_smooth = 0.1
        criterion = LabelSmoothCELoss(label_smooth, num_classes).cuda()
    elif data_name == "cifar10" or data_name == "cifar100":
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise NotImplementedError
    printRed(f"Loss: {criterion}")

    if args.distill:
        criterion_div = DistillKL(args.kd_T)
        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion)    # classification loss
        criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation

    # Optionally resume from a checkpoint
    if args.resume:
        printRed("Resuming from a checkpoint.")
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("  => loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("  => loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()


    # print("initial evaluation of the model")
    # validate(val_loader, model, criterion)
    
    # if args.quant:
    #     cali_loader = prepare_dataloader(traindir)
    #     enable_calibration(model)
    #     calibrate(cali_loader, model, args)
    #     enable_quantization(model)
    
    if args.evaluate:
        # new add for inference time
        # measure_inference(device, val_loader, model, criterion, val_loader_len, data_loader_type)

        # validate(val_loader, model, criterion) # org
        [prec1, prec5] = validate(val_loader, model, criterion, val_loader_len, data_loader_type)
        print(f'Top1-acc: {prec1}, Top5-acc: {prec5}', file=f, flush=True)
        if args.distill:
            printRed("Evaluate teacher:")
            [prec1, prec5] = validate(val_loader, module_list[-1], criterion, val_loader_len, data_loader_type)
        return

    total_time = AverageMeter() 
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

        # adjust lr, step
        # adjust_learning_rate(optimizer, epoch, args)

        # avg_train_time = train(train_loader, model, criterion, optimizer, epoch, tb_logger) # for step
        if args.distill:
            avg_train_time, model = train_distill(train_loader, module_list, criterion_list, optimizer, epoch, lr_scheduler, train_loader_len, data_loader_type, 
                                            val_loader, criterion, val_loader_len, # new add for evaluate test acc
                                                tb_logger, args) # for cosine
        else:
            avg_train_time = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, train_loader_len, data_loader_type, 
                                    val_loader, val_loader_len, # new add for evaluate test acc
                                    tb_logger) # for cosine
        # scheduler.step()
        total_time.update(avg_train_time)
        if args.test:
            break

        # evaluate on validation set
        [prec1, prec5] = validate(val_loader, model, criterion, val_loader_len, data_loader_type)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5) # new add
            lr = optimizer.param_groups[0]['lr']
            resLine = 'epoch: {}, lr: {:.5f}, prec1: {:.3f}, best prec1: {:.3f}, prec5: {:.3f}, best prec5: {:.3f}, avg_batch_time: {:.3f}'.format(epoch, lr, prec1, best_prec1, prec5, best_prec5, avg_train_time)
            print(resLine, file=f, flush=True)
            printRed(resLine)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5, # new add
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path=save_path)
            if epoch == args.epochs - 1:
                printRed('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(
                      prec1,
                      prec5,
                      args.total_batch_size / total_time.avg))
            # new add for tensorboard
            tb_logger.add_scalar('val_acc/acc1', prec1, epoch)
            tb_logger.add_scalar('val_acc/acc5', prec5, epoch)

        if data_loader_type == "dali":
            train_loader.reset()
            val_loader.reset()
        # print("Best prec1 : ", best_prec1)

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, train_loader_len, data_loader_type, 
            val_loader, val_loader_len,
            tb_logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    
    # lr_scheduler = init_lr_scheduler(optimizer)

    for i, data in enumerate(train_loader):
        
        if data_loader_type == "dali":
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        elif data_loader_type == "pytorch":
            input, target = data
            input = input.cuda()
            target = target.cuda()
        else:
            NotImplementedError


        # if i==0 and epoch==1:
        #     print("the very first calibration, happence only once")
        #     enable_calibration(model)
        #     calibrate(cali_loader, model, args)
        #     enable_quantization(model)

        curr_step = epoch * train_loader_len + i
        
        lr_scheduler.step(curr_step)

        # if epoch in [1,2,3] and i==0:
        if curr_step == 0:
            print("the very first calibration, happence only once")
            enable_calibration(model)
            with torch.no_grad():
                _ = model(input)
            # model.zero_grad()
            enable_quantization(model)

        # # lr_scheduler.get_lr()[0] is the main lr
        # current_lr = lr_scheduler.get_lr()[0] #org
        current_lr = optimizer.param_groups[0]['lr'] #new
        
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        # adjust_learning_rate_warm(optimizer, epoch, i, train_loader_len)
        if args.test:
            if i > 10:
                break

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
             loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            # new add to measure test accuracy
            # [prec1_test, prec5_test] = validate(val_loader, model, criterion, val_loader_len, data_loader_type)

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'LR {lr:.7f}'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5, lr=current_lr))
                # new
                tb_logger.add_scalar('train/acc1', prec1, curr_step)
                tb_logger.add_scalar('train/acc5', prec5, curr_step)
                tb_logger.add_scalar('train/lr_iteration', current_lr, curr_step)
                tb_logger.add_scalar('train/lr_epoch', current_lr, epoch)

                # new for evaluating 
                # tb_logger.add_scalar('val_acc/acc1_iteration', prec1_test, curr_step)
                # tb_logger.add_scalar('val_acc/acc5_iteration', prec5_test, curr_step)


        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

        # for test
        # if i == 2:
        #     break

    return batch_time.avg




def train_distill(train_loader, module_list, criterion_list, optimizer, epoch, lr_scheduler, train_loader_len, data_loader_type, 
                    val_loader, criterion, val_loader_len,
                        tb_logger=None, args=None):
    
    # ===> new add
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    # criterion_kd = criterion_list[2]

    model = module_list[0]
    model_t = module_list[-1]
    # ===> new add end

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
    
    # lr_scheduler = init_lr_scheduler(optimizer)

    for i, data in enumerate(train_loader):
        
        # input = data[0]["data"]
        # target = data[0]["label"].squeeze(-1).long()
        if data_loader_type == "dali":
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        elif data_loader_type == "pytorch":
            input, target = data
            input = input.cuda()
            target = target.cuda()
        else:
            NotImplementedError

        # if i==0 and epoch==1:
        #     print("the very first calibration, happence only once")
        #     enable_calibration(model)
        #     calibrate(cali_loader, model, args)
        #     enable_quantization(model)

        curr_step = epoch * train_loader_len + i
        
        # # print("cur:",curr_step)
        lr_scheduler.step(curr_step)

        # if epoch in [1,2,3] and i==0:
        if curr_step == 0:
            print("the very first calibration, happence only once")
            enable_calibration(model)
            with torch.no_grad():
                _ = model(input)
            # model.zero_grad()
            enable_quantization(model)

        # # lr_scheduler.get_lr()[0] is the main lr
        # current_lr = lr_scheduler.get_lr()[0] #org
        current_lr = optimizer.param_groups[0]['lr'] #new
        
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        # adjust_learning_rate_warm(optimizer, epoch, i, train_loader_len)
        if args.test:
            if i > 10:
                break

        # ===================forward=====================
        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        # new add
        with torch.no_grad():
            output_t = model_t(input)

        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # loss = criterion(output, target) # org
        # # new add, cls + kl div
        loss_cls = criterion_cls(output, target)
        loss_div = criterion_div(output, output_t)
        loss = args.gamma * loss_cls + args.alpha * loss_div   # + opt.beta * loss_kd
        if i == 0:
            print("\033[91m{}\033[00m" .format(f"gamma: {args.gamma}, alpha: {args.alpha}"))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # new add for release memory
        # gpu_usage()
        # del loss_cls, loss_div
        # del output_t
        # gc.collect()
        # torch.cuda.empty_cache()
        # gpu_usage()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
             loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            # new add to measure test accuracy 
            # [prec1_test, prec5_test] = validate(val_loader, model, criterion, val_loader_len, data_loader_type)

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'LR {lr:.7f}'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5, lr=current_lr))
                # new
                tb_logger.add_scalar('train/acc1', prec1, curr_step)
                tb_logger.add_scalar('train/acc5', prec5, curr_step)
                tb_logger.add_scalar('train/lr_iteration', current_lr, curr_step)
                tb_logger.add_scalar('train/lr_epoch', current_lr, epoch)

                # new for evaluating 
                # tb_logger.add_scalar('val_acc/acc1_iteration', prec1_test, curr_step)
                # tb_logger.add_scalar('val_acc/acc5_iteration', prec5_test, curr_step)


        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

        # for test
        # if i == 2:
        #     break

    return batch_time.avg, model




def validate(val_loader, model, criterion, val_loader_len, data_loader_type):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):

        # input = data[0]["data"]
        # target = data[0]["label"].squeeze(-1).long()
        # val_loader_len = int(val_loader._size / args.batch_size)
        if data_loader_type == "dali":
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
        elif data_loader_type == "pytorch":
            input, target = data
            input = input.cuda()
            target = target.cuda()
        else:
            NotImplementedError

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if args.local_rank == 0:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, save_path):
    torch.save(state, save_path+'/checkpoint.pth.tar')
    print(f"Save to: {save_path+'/checkpoint.pth.tar'}")
    if is_best:
        shutil.copyfile(save_path+'/checkpoint.pth.tar', save_path+'/model_best.pth.tar')
        print(f"Save best to: {save_path+'/model_best.pth.tar'}")



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


def adjust_learning_rate_warm(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def init_lr_scheduler(optimizer):
    # base_lr==: 0.0001
    # warmup_lr: 0.004
    # warmup_steps: 2500
    # max_iter: 250000
    # min_lr: 0.0 
    return Cosine(optimizer, max_iter = 250000, min_lr = 0.0, base_lr = 0.0001, warmup_lr = 0.004, warmup_steps = 2500)

    # below is the setup adopted to for single gpu apex case
    # return Cosine(optimizer, max_iter=500500, min_lr=0.0, base_lr=0.0001, warmup_lr=0.004/(2**0.5), warmup_steps=5005)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def calibrate(cali_loader, model, args):
    model.eval()
    print("Start calibration ...")
    print("Calibrate images number = ", len(cali_loader.dataset))
    with torch.no_grad():
        for i, (images, target) in enumerate(cali_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            output = model(images)
            print("Calibration ==> ", i+1)
    print("End calibration.")
    return


def prepare_dataloader(traindir):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    cali_batch_size = 10
    cali_batch = 10
    cali_dataset = torch.utils.data.Subset(train_dataset, indices=torch.arange(cali_batch_size * cali_batch))
    cali_loader = torch.utils.data.DataLoader(cali_dataset, batch_size=cali_batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    return cali_loader


if __name__ == '__main__':
    main()
