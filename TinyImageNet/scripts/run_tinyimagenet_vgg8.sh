# # conda activate mqbench
# #!/bin/bash

# set -e
# # make the script exit with an error whenever an error occurs (and is not explicitly handled).

# # start timing
# start=$(date +%s)
# start_fmt=$(date +%Y-%m-%d\ %r)
# echo "STARTING TIMING RUN AT $start_fmt"


# METHOD_TYPE=$1
# echo $METHOD_TYPE


# # ====================== Tiny-ImageNet, VGG-8 =====================================
# # Note: whether to add the following to the model:
# # if h == 64:
#     # x = self.pool3(x)
# # ******************************************************************************************
# # SGD
# # batch-size 
#     # FP: 32
#     # PACT: 32
#     # LSQ, DoReFa: 
# # lr, weight_decay, initilization
#     # fp: 
#     # 4-bit Quantized: 
#     # 1-bit and 2-bit Quantized: 
# # ========================================================================================


# # # ====================================== FP, PACT, W4A4 =============================================
# # # ===== fp
# if [ $METHOD_TYPE == "vgg8_bn_fp_lr0.004_wd1e-4" ] 
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg8_bn/$METHOD_TYPE" \
#     -a vgg8_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 0.004 \
#     --weight_decay 1e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --not-quant \
#     --epochs=100


# elif [ $METHOD_TYPE == "vgg8_bn_fp_lr5e-2_wd5e-4" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg8_bn/$METHOD_TYPE" \
#     -a vgg8_bn \
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

# # teacher is vgg8_bn_fp_lr5e-2_wd5e-4
# # ===== Pact
# elif [ $METHOD_TYPE == "pact_overall/vgg8_bn_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain_teacher45.06" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg8_bn/$METHOD_TYPE" \
#     -a vgg8_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg8_bn/vgg8_bn_fp_lr5e-2_wd5e-4/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== Pact + KD
# elif [ $METHOD_TYPE == "pact_overall/vgg8_bn_pact_a4w4_kd_gamma0.0_alpha1.0_sgd_lr5e-4_wd5e-4_initPretrain_teacher45.06" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg8_bn/$METHOD_TYPE" \
#     -a vgg8_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg8_bn/vgg8_bn_fp_lr5e-2_wd5e-4/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch vgg8_bn \
#     --teacher_path "./results/tiny-imagenet_vgg8_bn/vgg8_bn_fp_lr5e-2_wd5e-4/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100


# # teacher is vgg8_bn_fp_lr0.004_wd1e-4
# # ===== Pact
# elif [ $METHOD_TYPE == "pact_overall/vgg8_bn_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain_teacher49.58" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg8_bn/$METHOD_TYPE" \
#     -a vgg8_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg8_bn/vgg8_bn_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
#     --epochs=100


# # ===== Pact + KD
# elif [ $METHOD_TYPE == "pact_overall/vgg8_bn_pact_a4w4_kd_gamma0.0_alpha1.0_sgd_lr5e-4_wd5e-4_initPretrain_teacher49.58" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_vgg8_bn/$METHOD_TYPE" \
#     -a vgg8_bn \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --load_pretrain \
#     --pretrain_path "./results/tiny-imagenet_vgg8_bn/vgg8_bn_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
#     --distill True \
#     --teacher_arch vgg8_bn \
#     --teacher_path "./results/tiny-imagenet_vgg8_bn/vgg8_bn_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 1.0 \
#     --epochs=100


# fi
# # # ========================================================================================




# # end timing
# end=$(date +%s)
# end_fmt=$(date +%Y-%m-%d\ %r)
# echo "ENDING TIMING RUN AT $end_fmt"

# # report result
# total_time=$(( $end - $start ))

# echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"

# # choices=["org","STE","EWGS","Uscheduler","Uscheduler_Pscheduler"]
# # resnet18_4bit_pact_4gpus_bs128_org
# # --opt-level O1 \
# # CUDA_VISIBLE_DEVICES=0,1,2,3 
# # /home/ubuntu/data/tiny-imagenet-200 \
# # efficientnet_b0