#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
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



# EWGS+SQAKD
if [ $METHOD_TYPE == "EWGS+SQAKD/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0 \
                        --epochs 1200 


# EWGS+AT
elif [ $METHOD_TYPE == "EWGS+AT/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'attention' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1000 \
                        --epochs 1200



# EWGS+NST
elif [ $METHOD_TYPE == "EWGS+NST/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'nst' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 50 \
                        --epochs 1200


# EWGS+SP
elif [ $METHOD_TYPE == "EWGS+SP/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'similarity' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 3000 \
                        --epochs 1200


# EWGS+RKD
elif [ $METHOD_TYPE == " EWGS+RKD/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'rkd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1 \
                        --epochs 1200


# EWGS+CRD
elif [ $METHOD_TYPE == "EWGS+CRD/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.8 \
                        --epochs 1200


# EWGS+FitNet
elif [ $METHOD_TYPE == "EWGS+FitNet/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'hint' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 100 \
                        --epochs 1200


# EWGS+CC
elif [ $METHOD_TYPE == "EWGS+CC/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'correlation' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.02 \
                        --epochs 1200


# EWGS+VID
elif [ $METHOD_TYPE == "EWGS+VID/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'vid' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0 \
                        --epochs 1200


# EWGS+FSP
elif [ $METHOD_TYPE == "EWGS+FSP/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'fsp' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 50 \
                        --epochs 1200

# EWGS+FT
elif [ $METHOD_TYPE == "EWGS+FT/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'factor' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 200 \
                        --epochs 1200


# EWGS+CKTF
elif [ $METHOD_TYPE == "EWGS+CKTF/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --distill 'crdst' \
                        --teacher_arch 'resnet20_fp' \
                        --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0 \
                        --kd_theta 0.8 \
                        --nce_k 4096 \
                        --epochs 1200
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"