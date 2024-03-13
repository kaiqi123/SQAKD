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


# ====================== Tiny-ImageNet, VGG-11 =====================================
# Note: whether to add the following to the model:
# if h == 64:
    # x = self.pool3(x)
# ******************************************************************************************
# SGD
# batch-size: 32
# lr, weight_decay, initilization 
#     fp: lr: 5e-2, weight_decay: 5e-4 (same as resnet-18)
#     4-bit Quantized: 5e-4, weight_decay: 5e-4, with initilized with pretrained fp model (same as resnet-18)
#     3-bit Quantized: lr: 0.004, weight_decay: 1e-4
#                 PACT: vgg11_bn_pact_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain
#                 PACT + KD: vgg11_bn_pact_a3w3_kd_gamma0.0_alpha10.0_sgd_lr0.004_wd1e-4_withoutPretrain 
# ========================================================================================


# # ====================================== FP, PACT, W4A4 =============================================
# # ===== fp
# if [ $METHOD_TYPE == "vgg11_bn_fp_lr5e-2_wd5e-4" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-2 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --not-quant \
#     --epochs=100

# # ===== Pact
# elif [ $METHOD_TYPE == "pact_overall/vgg11_bn_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # # ===== Pact + KD
# elif [ $METHOD_TYPE == "pact_overall/vgg11_bn_pact_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
    # CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
    # --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
    # -a vgg11_bn \
    # --batch-size 32 \
    # --loss-scale 128.0 \
    # --workers 10 \
    # --optimizer_type 'SGD' \
    # --lr 5e-4 \
    # --weight_decay 5e-4 \
    # --backward_method "org" \
    # /home/users/kzhao27/tiny-imagenet-200 \
    # --load_pretrain \
    # --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
    # --distill True \
    # --teacher_arch vgg11_bn \
    # --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
    # --gamma 0.0 \
    # --alpha 100.0 \
    # --epochs=100
# fi
# # ========================================================================================


# # # ====================================== PACT, W3A3 =============================================
# # PACT
# if [ $METHOD_TYPE == "pact_overall/a3w3/vgg11_bn_pact_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100

# # PACT + KD
# elif [ $METHOD_TYPE == "pact_overall/a3w3/vgg11_bn_pact_a3w3_kd_gamma0.0_alpha10.0_sgd_lr0.004_wd1e-4_withoutPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 10.0 \
#     --epochs=100
# fi
# # ========================================================================================



# # ====================================== FP, PACT, W8A8 =============================================


# # ===== Pact
# if [ $METHOD_TYPE == "pact_overall/a8w8/vgg11_bn_pact_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== Pact + KD
# elif [ $METHOD_TYPE == "pact_overall/a8w8/vgg11_bn_pact_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 100.0 \
#     --epochs=100

# # ===== LSQ
# elif [ $METHOD_TYPE == "lsq_overall/a8w8/vgg11_bn_lsq_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== LSQ + KD
# elif [ $METHOD_TYPE == "lsq_overall/a8w8/vgg11_bn_lsq_a4w4_kd_gamma0.0_alpha200.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 200.0 \
#     --epochs=100

# # ===== Dorefa
# elif [ $METHOD_TYPE == "dorefa_overall/a8w8/vgg11_bn_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== Dorefa + KD
# elif [ $METHOD_TYPE == "dorefa_overall/a8w8/vgg11_bn_dorefa_a8w8_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 100.0 \
#     --epochs=100
# fi
# # ========================================================================================



# # # ====================================== PACT, W3A3, different backwards (Need to modify code a little bit, maybe out of memory) =============================================
# # PACT (forward: PACT, backward: EWGS) + KD
# # Change --backward_method from "org" to "EWGS"
# # ======================================================================================================================
# # if [ $METHOD_TYPE == "pact_overall/a3w3/vgg11_bn_forward:pact_backward:EWGS_a3w3_kd_gamma0.0_alpha10.0_sgd_lr0.004_wd1e-4_withoutPretrain" ] 
# if [ $METHOD_TYPE == "test" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 10.0 \
#     --epochs=100
# fi
# # ========================================================================================



# ====================================== W4A4, Effect of initialization ================================================
# Rember to modify get_config.py!!!!!!!
# Random initialization, Remove: 
    # --load_pretrain \
    # --pretrain_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
# *********************************************************************************************************************
# # ===== Pact
# if [ $METHOD_TYPE == "effect_of_initialization/pact/vgg11_bn_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --epochs=100


# # # ===== Pact + KD
# elif [ $METHOD_TYPE == "effect_of_initialization/pact/vgg11_bn_pact_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 100.0 \
#     --epochs=100

# # ===== LSQ
# elif [ $METHOD_TYPE == "effect_of_initialization/lsq/vgg11_bn_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --epochs=100


# # # ===== LSQ + KD
# elif [ $METHOD_TYPE == "effect_of_initialization/lsq/vgg11_bn_lsq_a4w4_kd_gamma0.0_alpha200.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 200.0 \
#     --epochs=100


# # ===== dorefa
# elif [ $METHOD_TYPE == "effect_of_initialization/dorefa/vgg11_bn_dorefa_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --epochs=100


# # # ===== dorefa + KD
# elif [ $METHOD_TYPE == "effect_of_initialization/dorefa/vgg11_bn_dorefa_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
#     -a vgg11_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch vgg11_bn \
#     --teacher_path "./results/tiny-imagenet_vgg11_bn/vgg11_bn_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 100.0 \
#     --epochs=100
# fi


# # c4n4
# # ===== LSQ
if [ $METHOD_TYPE == "effect_of_initialization/lsq/vgg11_bn_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain_t2" ]
then
    CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
    -a vgg11_bn \
    --batch-size 32 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --epochs=100

elif [ $METHOD_TYPE == "effect_of_initialization/lsq/vgg11_bn_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain_t3" ]
then
    CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
    -a vgg11_bn \
    --batch-size 32 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --epochs=100

elif [ $METHOD_TYPE == "test" ]
then
    CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_vgg11_bn/$METHOD_TYPE" \
    -a vgg11_bn \
    --batch-size 32 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --epochs=100
fi
# # ========================================================================================



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