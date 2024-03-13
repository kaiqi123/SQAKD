# conda activate mqbench
#!/bin/bash

set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE

# ====================== Tiny-ImageNet, ResNet-18 ==========================================
# SGD
# batch-size 64
# lr, weight_decay, initilization
    # fp: lr: 5e-2, weight_decay: 5e-4
    # 4-bit Quantized: lr: 5e-4, weight_decay: 5e-4, with initilized with pretrained fp model;
    # 3-bit Quantized: lr: 0.004, weight_decay: 1e-4, with initilized with pretrained fp model;    
    # 1-bit and 2-bit Quantized, not stable
        # pact,  lr0.004, weight_decay 1e-4, without pretrain, acc: 33.2
        # pact+kd, lr0.008, weight_decay 1e-4, without pretrain, acc: 37.900
# ========================================================================================


# # ====================================== W4A4, FP, PACT =============================================
# # ===== fp,sgd_lr5e-2_wd5e-4
# if [ $METHOD_TYPE == "resnet18_fp" ] 
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-2 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --not-quant \
#     --epochs=100



# # ===== pact, sgd_lr5e-4_wd5e-4_initPretrain
# elif [ $METHOD_TYPE == "pact_overall/resnet18_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain_t2" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100

# # ===== Pact + KD, sgd_lr5e-4_wd5e-4_initPretrain, Loss = 0.7 * Lce + 0.3 * Lkd
# elif [ $METHOD_TYPE == "pact_overall/resnet18_pact_a4w4_kd_gamma0.7_alpha0.3_sgd_lr5e-4_wd5e-4_initPretrain_t2" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.7 \
#     --alpha 0.3 \
#     --epochs=100

# fi
# # ========================================================================================




# ================================================ A3W3, PACT, LSQ, Dorefa ==========================================
# 3 bit
# lr0.004, weight_decay 1e-4, with pretrained model
# pact, acc: 58.09
# pact+kd, acc: 61.34
# ******************************************************************************************************************** 

# # ===== Pact
# # with pretrained model
# if [ $METHOD_TYPE == "pact_overall/pact_a3w3/resnet18_pact_a3w3_independent_sgd_lr0.004_wd1e-4_withPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100

# # ===== Pact + KD, Loss = Lkd
# elif [ $METHOD_TYPE == "pact_overall/pact_a3w3/resnet18_pact_a3w3_kd_gamma0.0_alpha1.0_sgd_lr0.004_wd1e-4_withPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100

# # ===== LSQ 
# elif [ $METHOD_TYPE == "lsq_overall/resnet18_lsq_a3w3_independent_sgd_lr0.004_wd1e-4_withPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100

# # ===== LSQ + KD, Loss = Lkd
# elif [ $METHOD_TYPE == "lsq_overall/resnet18_lsq_a3w3_kd_gamma0.0_alpha1.0_sgd_lr0.004_wd1e-4_withPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100

# # ===== Dorefa
# elif [ $METHOD_TYPE == "dorefa_overall/resnet18_dorefa_a3w3_independent_sgd_lr0.004_wd1e-4_withPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100

# # ===== Dorefa + KD, Loss = Lkd
# elif [ $METHOD_TYPE == "dorefa_overall/resnet18_dorefa_a3w3_kd_gamma0.0_alpha1.0_sgd_lr0.004_wd1e-4_withPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100
# fi
# # =========================================================================================================================




# # ====================================== W8A8, PACT, LSQ, Dorefa =============================================
# Same as 4-bit Quantized: lr: 
# 5e-4, weight_decay: 5e-4, with initilized with pretrained fp mode, SGD;
# ********************************************************************************************************************

# # ===== PACT
# if [ $METHOD_TYPE == "pact_overall/pact_a8w8/resnet18_pact_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== Pact + KD
# elif [ $METHOD_TYPE == "pact_overall/pact_a8w8/resnet18_pact_a8w8_kd_gamma0_alpha1_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100


# # ===== LSQ
# elif [ $METHOD_TYPE == "lsq_overall/resnet18_lsq_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== LSQ + KD
# elif [ $METHOD_TYPE == "lsq_overall/resnet18_lsq_a8w8_kd_gamma0_alpha1_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100

# # ===== Dorefa
# elif [ $METHOD_TYPE == "dorefa_overall/resnet18_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== Dorefa + KD
# elif [ $METHOD_TYPE == "dorefa_overall/resnet18_dorefa_a8w8_kd_gamma0_alpha1_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100

# fi
# # ==================================================================================================================



# ====================================== W4A4, Effect of initialization (The results of SQAKD are not good) ================================================
# Rember to modify get_config.py!!!!!!!
# Random initialization, Remove: 
    # --load_pretrain \
    # --pretrain_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
# *********************************************************************************************************************
# # ===== pact, sgd_lr5e-4_wd5e-4_initPretrain
# if [ $METHOD_TYPE == "effect_of_initialization/pact/resnet18_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --epochs=100

# # ===== Pact + KD, sgd_lr5e-4_wd5e-4_initPretrain
# elif [ $METHOD_TYPE == "effect_of_initialization/pact/resnet18_pact_a4w4_kd_gamma0_alpha1_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100

# # ===== lsq, sgd_lr5e-4_wd5e-4_initPretrain
# elif [ $METHOD_TYPE == "effect_of_initialization/lsq/resnet18_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --epochs=100

# # ===== lsq + KD, sgd_lr5e-4_wd5e-4_initPretrain
# elif [ $METHOD_TYPE == "effect_of_initialization/lsq/resnet18_lsq_a4w4_kd_gamma0_alpha1_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100

# # ===== dorefa, sgd_lr5e-4_wd5e-4_initPretrain
# elif [ $METHOD_TYPE == "effect_of_initialization/dorefa/resnet18_dorefa_a4w4_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10  \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --backward_method "org" \
#     --epochs=100

# # ===== dorefa + KD, sgd_lr5e-4_wd5e-4_initPretrain
# elif [ $METHOD_TYPE == "effect_of_initialization/dorefa/resnet18_dorefa_a4w4_kd_gamma0_alpha1_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_resnet18/$METHOD_TYPE" \
#     -a resnet18_imagenet \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch resnet18_imagenet \
#     --teacher_path "./results/tiny-imagenet_resnet18/resnet18_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100
# fi
# # ==================================================================================================================

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"

# choices=["org","STE","EWGS","Uscheduler","Uscheduler_Pscheduler"]
# resnet18_4bit_pact_4gpus_bs128_org
# --opt-level O1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 
# /home/ubuntu/data/tiny-imagenet-200 \
# efficientnet_b0

