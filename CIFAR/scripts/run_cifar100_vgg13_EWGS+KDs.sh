#!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: VGG-13
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: EWGS + other distillation
# Bit-width: W2A2
# EWGS + SQAKD: wihout labels (gammaa=0.0)
# EWGS + other distillation: with labels (gamma = 1.0)
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE



# EWGS + SQAKD
if [ $METHOD_TYPE == "EWGS+SQAKD/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0



# EWGS + AT
elif [ $METHOD_TYPE == "EWGS+AT/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'attention' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1000

# EWGS + NST
elif [ $METHOD_TYPE == "EWGS+NST/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'nst' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 50

# EWGS + SP
elif [ $METHOD_TYPE == "EWGS+SP/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'similarity' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 3000


# EWGS + RKD
elif [ $METHOD_TYPE == "EWGS+RKD/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'rkd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1


# EWGS + CRD
elif [ $METHOD_TYPE == "EWGS+CRD/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.8


# EWGS + FitNet
elif [ $METHOD_TYPE == "EWGS+FitNet/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'hint' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 100


# EWGS + CC
elif [ $METHOD_TYPE == "EWGS+CC/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'correlation' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.02

# EWGS + VID
elif [ $METHOD_TYPE == "EWGS+VID/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'vid' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0

# EWGS + FSP
elif [ $METHOD_TYPE == "EWGS+FSP/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'fsp' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 50

# EWGS + FT
elif [ $METHOD_TYPE == "EWGS+FT/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'factor' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 200


# EWGS + CKTF
elif [ $METHOD_TYPE == "EWGS+CKTF/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crdst' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0 \
                        --kd_theta 0.8 \
                        --nce_k 4096
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"
