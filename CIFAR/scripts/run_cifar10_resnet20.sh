#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP, EWGS, EWGS+SQAKD
# Bit-width: W1A1, W2A2, W4A4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE



if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --epochs 1200

# W1A1
# EWGS
elif [ $METHOD_TYPE == "EWGS/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --epochs 1200

# W1A1
# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --weight_levels 2 \
                        --act_levels 2 \
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

# W2A2
# EWGS
elif [ $METHOD_TYPE == "EWGS/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --epochs 1200

# W2A2
# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '3' \
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

# W4A4
# EWGS
elif [ $METHOD_TYPE == "EWGS/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                        --epochs 1200

# W4A4
# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --weight_levels 16 \
                        --act_levels 16 \
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
 
fi




# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"


