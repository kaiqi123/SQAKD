#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: VGG-8
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP, EWGS, EWGS+SQAKD
# Bit-width: W2A2
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE


# FP
if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --arch 'vgg8_bn_fp' \
                    --log_dir './results/CIFAR10_VGG8/'$METHOD_TYPE

# EWGS
elif [ $METHOD_TYPE == "EWGS/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --arch 'vgg8_bn_quant' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_VGG8/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_VGG8/'$METHOD_TYPE \
                        --epochs 400

# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --arch 'vgg8_bn_quant' \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR10_VGG8/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR10_VGG8/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg8_bn_fp' \
                        --teacher_path './results/CIFAR10_VGG8/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0 \
                        --epochs 400

fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"