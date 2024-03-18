#!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: ResNet-32
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


# ===========================================
# cifar100
# resnet32
# ===========================================

          

# ======================================================= FP, W4A4, W2A2, W1A1, W3A3 ===============================================================
if [ $METHOD_TYPE == "720epocchs/fp_crd_cosine/" ] 
then
    python3 train_fp.py --gpu_id '3' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE

# === W2A2
# EWGS_hess, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
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
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE



# EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha1_beta0/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '3' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
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
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet32_fp' \
                        --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0


# === W1A1
# EWGS_hess, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
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
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE



# EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha1_beta0/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
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
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet32_fp' \
                        --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0

# === W4A4
# EWGS_hess, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
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
                        --weight_levels 16 \
                        --act_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE



# EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha100_beta0/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '2' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
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
                        --weight_levels 16 \
                        --act_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet32_fp' \
                        --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 100.0 \
                        --kd_beta 0.0
fi
# =============================================================================================================



# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"