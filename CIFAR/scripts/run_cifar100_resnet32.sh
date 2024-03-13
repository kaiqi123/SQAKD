#!/bin/bash


####################################################################################
# You may run the code using the following commands.
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.
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

# ===========================================fp, EWGS_hess, and EWGS_hess + KD ===========================================================
# # fp, Follow CRD settings 
# if [ $METHOD_TYPE == "fp_crd_step/" ] 
# then
#     python3 train_fp.py --gpu_id '1' \
#                     --dataset 'cifar100' \
#                     --arch 'resnet32_fp' \
#                     --num_workers 8 \
#                     --batch_size 64 \
#                     --epochs 240 \
#                     --lr_m 0.05 \
#                     --weight_decay 5e-4 \
#                     --lr_scheduler_m 'step' \
#                     --decay_schedule_m '150-180-210' \
#                     --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE

# # fp, Follow CRD settings but with cosine learning rate
# elif [ $METHOD_TYPE == "fp_crd_cosine/" ] 
# then
#     python3 train_fp.py --gpu_id '2' \
#                     --dataset 'cifar100' \
#                     --arch 'resnet32_fp' \
#                     --num_workers 8 \
#                     --batch_size 64 \
#                     --epochs 240 \
#                     --lr_m 0.05 \
#                     --weight_decay 5e-4 \
#                     --lr_scheduler_m 'cosine' \
#                     --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE

# # EWGS_hess, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "EWGS_hess_Adam_lrm5e-4_lrq5e-6/W2A2/" ] 
# then
#     python3 train_quant.py --gpu_id '0' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --epochs 240 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --weight_levels 4 \
#                         --act_levels 4 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/EWGS_hess_Adam_lrm5e-4_lrq5e-6/'$METHOD_TYPE



# # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha1_beta0/W2A2/" ] 
# then
#     python3 train_quant.py --gpu_id '1' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --epochs 240 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --weight_levels 4 \
#                         --act_levels 4 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/EWGS_hess_Adam_lrm5e-4_lrq5e-6/'$METHOD_TYPE \
#                         --distill 'kd' \
#                         --teacher_arch 'resnet32_fp' \
#                         --teacher_path '../results/CIFAR100_ResNet32/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --kd_gamma 0.0 \
#                         --kd_alpha 1.0 \
#                         --kd_beta 0.0
# =============================================================================================================================
          

# ======================================================= 720 epochs, W4A4, W2A2, W1A1, W3A3 ===============================================================
# # fp, Follow CRD settings but with cosine learning rate
# if [ $METHOD_TYPE == "720epocchs/fp_crd_cosine/" ] 
# then
#     python3 train_fp.py --gpu_id '3' \
#                     --dataset 'cifar100' \
#                     --arch 'resnet32_fp' \
#                     --num_workers 8 \
#                     --batch_size 64 \
#                     --lr_m 0.05 \
#                     --weight_decay 5e-4 \
#                     --lr_scheduler_m 'cosine' \
#                     --epochs 720 \
#                     --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE

# # === W2A2
# # EWGS_hess, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6/W2A2/" ] 
# then
#     python3 train_quant.py --gpu_id '2' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 4 \
#                         --act_levels 4 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE



# # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha1_beta0/W2A2/" ] 
# then
#     python3 train_quant.py --gpu_id '3' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 4 \
#                         --act_levels 4 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
#                         --distill 'kd' \
#                         --teacher_arch 'resnet32_fp' \
#                         --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --kd_gamma 0.0 \
#                         --kd_alpha 1.0 \
#                         --kd_beta 0.0


# # === W1A1
# # EWGS_hess, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6/W1A1/" ] 
# then
#     python3 train_quant.py --gpu_id '0' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 2 \
#                         --act_levels 2 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE



# # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha1_beta0/W1A1/" ] 
# then
#     python3 train_quant.py --gpu_id '1' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 2 \
#                         --act_levels 2 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
#                         --distill 'kd' \
#                         --teacher_arch 'resnet32_fp' \
#                         --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --kd_gamma 0.0 \
#                         --kd_alpha 1.0 \
#                         --kd_beta 0.0

# # === W4A4
# # EWGS_hess, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6/W4A4/" ] 
# then
#     python3 train_quant.py --gpu_id '2' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 16 \
#                         --act_levels 16 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE



# # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha100_beta0/W4A4/" ] 
# then
#     python3 train_quant.py --gpu_id '2' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 16 \
#                         --act_levels 16 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
#                         --distill 'kd' \
#                         --teacher_arch 'resnet32_fp' \
#                         --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --kd_gamma 0.0 \
#                         --kd_alpha 100.0 \
#                         --kd_beta 0.0

# ===== W3A3
# # # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
# elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha200_beta0/W3A3/" ] 
# then
#     python3 train_quant.py --gpu_id '0' \
#                         --dataset 'cifar100' \
#                         --arch 'resnet32_quant' \
#                         --num_workers 8 \
#                         --batch_size 64 \
#                         --weight_decay 5e-4 \
#                         --optimizer_m 'Adam' \
#                         --optimizer_q 'Adam' \
#                         --lr_m 5e-4 \
#                         --lr_q 5e-6 \
#                         --lr_scheduler_m 'cosine' \
#                         --lr_scheduler_q 'cosine' \
#                         --epochs 720 \
#                         --weight_levels 8 \
#                         --act_levels 8 \
#                         --baseline False \
#                         --use_hessian True \
#                         --load_pretrain True \
#                         --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
#                         --distill 'kd' \
#                         --teacher_arch 'resnet32_fp' \
#                         --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
#                         --kd_gamma 0.0 \
#                         --kd_alpha 200.0 \
#                         --kd_beta 0.0

# ======  --weight_decay 25e-6
# ===== W3A3
# c4n5
# # # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
if [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha700_beta0_wd25e-6/W3A3/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 25e-6 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --weight_levels 8 \
                        --act_levels 8 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --log_dir '../results/CIFAR100_ResNet32/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'resnet32_fp' \
                        --teacher_path '../results/CIFAR100_ResNet32/720epochs/fp_crd_cosine/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 700.0 \
                        --kd_beta 0.0

# ===== W4A4
# # # EWGS_hess + KD, Adam_lrm5e-4_lrq5e-6
elif [ $METHOD_TYPE == "720epochs/EWGS_hess_Adam_lrm5e-4_lrq5e-6_kd_gamma0_alpha100_beta0_wd25e-6/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '1' \
                        --dataset 'cifar100' \
                        --arch 'resnet32_quant' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 25e-6 \
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