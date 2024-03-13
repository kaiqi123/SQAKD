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


# # ====================== Tiny-ImageNet, SqueezeNet1_0 =====================================
# # SGD
# # batch-size: 64
# # lr, weight_decay, initilization
# #     fp: lr: 5e-2, weight_decay: 5e-4 (same as resnet-18), acc: 51.49
# #     4-bit and 8-bit Quantization: 5e-4, weight_decay: 5e-4, with initilized with pretrained fp model (same as resnet-18)
# #         LSQ/Dorefa: squeezenet_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain
# #         LSQ/Dorefa + KD: alpha 50, squeezenet_pact_a4w4_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain
# # ========================================================================================



# # ====================================== FP, PACT, W4A4 =============================================
# # ===== fp

# # if [ $METHOD_TYPE == "squeezenet_fp" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-2 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --not-quant \
# #     --epochs=100


# # # ===== Dorefa, W4A4
# # elif [ $METHOD_TYPE == "dorefa_overall/a4w4/squeezenet_dorefa_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Dorefa + KD, W4A4
# # elif [ $METHOD_TYPE == "dorefa_overall/a4w4/squeezenet_dorefa_a4w4_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch squeezenet1_0 \
# #     --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 50.0 \
# #     --epochs=100


# # ===== Dorefa, W8A8
# # elif [ $METHOD_TYPE == "dorefa_overall/a8w8/squeezenet_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Dorefa + KD, W8A8
# # elif [ $METHOD_TYPE == "dorefa_overall/a8w8/squeezenet_dorefa_a8w8_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch squeezenet1_0 \
# #     --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 50.0 \
# #     --epochs=100


# # # ===== LSQ, W4A4
# # elif [ $METHOD_TYPE == "lsq_overall/a4w4/squeezenet_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== LSQ + KD, W4A4
# # elif [ $METHOD_TYPE == "lsq_overall/a4w4/squeezenet_lsq_a4w4_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch squeezenet1_0 \
# #     --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 50.0 \
# #     --epochs=100


# # # ===== LSQ, W8A8
# # elif [ $METHOD_TYPE == "lsq_overall/a8w8/squeezenet_lsq_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== LSQ + KD, W8A8
# # elif [ $METHOD_TYPE == "lsq_overall/a8w8/squeezenet_lsq_a8w8_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
# #     -a squeezenet1_0 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch squeezenet1_0 \
# #     --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 50.0 \
# #     --epochs=100
# # fi
# # ========================================================================================



# # ====================================== Dorefa, W8A8, Effect of initialization (Results are not good without initing from teacher) ================================================
# # # Random initialization, Remove: 
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
# # # ******************************************************************************************************************************
# # ===== Dorefa, W8A8
# if [ $METHOD_TYPE == "dorefa_effect_of_initialization/a8w8/squeezenet_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
#     -a squeezenet1_0 \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --epochs=100


# # ===== Dorefa + KD, W8A8
# elif [ $METHOD_TYPE == "dorefa_effect_of_initialization/a8w8/squeezenet_dorefa_a8w8_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
#     -a squeezenet1_0 \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch squeezenet1_0 \
#     --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 50.0 \
#     --epochs=100
# fi
# # ======================================================================================================================



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