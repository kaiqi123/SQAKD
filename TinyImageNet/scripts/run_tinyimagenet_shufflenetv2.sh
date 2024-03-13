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


# # ====================== Tiny-ImageNet, shufflenetV2 =====================================
# # SGD
# # batch-size: 64
# # lr, weight_decay, initilization
#     # fp: lr: 5e-2, weight_decay: 5e-4 (same as resnet-18), acc: 49.91
#     # 4-bit Quantized: 5e-4, weight_decay: 5e-4, with initilized with pretrained fp model (same as resnet-18)
#     #         PACT: mobilenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain
#     #         PACT + KD: alpha 100, mobilenetV2_pact_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain
#     # 8-bit Quantized: 5e-4, weight_decay: 5e-4, with initilized with pretrained fp model
#     #         Dorefa: shufflenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain
#     #         Dorefa + KD: shufflenetV2_dorefa_a8w8_kd_gamma0.0_alpha300.0_sgd_lr5e-4_wd5e-4_initPretrain
# # ========================================================================================



# # ====================================== FP, W4A4 =============================================

# # ===== fp
# # if [ $METHOD_TYPE == "shufflenetV2_fp" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
# #     -a shufflenet_v2_x0_5 \
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

# # # ===== Pact, W4A4
# # elif [ $METHOD_TYPE == "pact_overall/a4w4/shufflenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
# #     -a shufflenet_v2_x0_5 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Pact + KD, W4A4
# # elif [ $METHOD_TYPE == "pact_overall/a4w4/shufflenetV2_pact_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
# #     -a shufflenet_v2_x0_5 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch shufflenet_v2_x0_5 \
# #     --teacher_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100
# # fi 
# # =====================================================================================================================




# # ====================================================== W8A8 ========================================================
# # ===== Dorefa, W8A8
# # if [ $METHOD_TYPE == "dorefa_overall/a8w8/shufflenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
# #     -a shufflenet_v2_x0_5 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Dorefa + KD, W8A8
# # if [ $METHOD_TYPE == "dorefa_overall/a8w8/shufflenetV2_dorefa_a8w8_kd_gamma0.0_alpha800.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
# #     -a shufflenet_v2_x0_5 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch shufflenet_v2_x0_5 \
# #     --teacher_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 800.0 \
# #     --epochs=100
# # fi 
# # ========================================================================================



# # # ====================================== PACT, W4A4, different backwards =============================================
# # PACT (forward: PACT, backward: EWGS) + KD
# # Change --backward_method from "org" to "EWGS"
# # ======================================================================================================================
# # ===== PACT (forward: PACT, backward: EWGS) + KD, W4A4
# # if [ $METHOD_TYPE == "pact_overall/a4w4/shufflenetV2_f:pact_b:ewgs_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
# #     -a shufflenet_v2_x0_5 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "EWGS" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch shufflenet_v2_x0_5 \
# #     --teacher_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100
# # fi
# # ======================================================================================================================



# # ====================================== Dorefa, W8A8, Effect of initialization (Results are not good without initing from teacher) ================================================
# # # Random initialization, Remove: 
#     # --load_pretrain \
#     # --pretrain_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
# # # ******************************************************************************************************************************

# if [ $METHOD_TYPE == "dorefa_effect_of_initialization/a8w8/shufflenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
#     -a shufflenet_v2_x0_5 \
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
# elif [ $METHOD_TYPE == "dorefa_effect_of_initialization/a8w8/shufflenetV2_dorefa_a8w8_kd_gamma0.0_alpha800.0_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_shufflenetV2/$METHOD_TYPE" \
#     -a shufflenet_v2_x0_5 \
#     --batch-size 64 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch shufflenet_v2_x0_5 \
#     --teacher_path "./results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 800.0 \
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