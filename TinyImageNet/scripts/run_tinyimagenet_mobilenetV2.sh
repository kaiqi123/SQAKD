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


# # ====================== Tiny-ImageNet, MobileNet-V2 =====================================
# # SGD
# # batch-size 
# #     PACT: 64
# #     LSQ, DoReFa: 32
# # lr, weight_decay, initilization
# #     fp: lr: 0.004, weight_decay: 1e-4, 58.07
# #     4-bit and 8-bit Quantized: 5e-4, weight_decay: 5e-4, with initilized with pretrained fp model (same as resnet-18)
# #         PACT/LSQ/Dorefa: mobilenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain
# #         PACT/LSQ/Dorefa + KD: mobilenetV2_pact_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain
# #     3-bit Quantized: lr: 0.004, weight_decay: 1e-4, with initilized with pretrained fp model
# #         PACT/LSQ/Dorefa: mobilenetV2_pact_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain    
# #         PACT/LSQ/Dorefa + KD: mobilenetV2_pact_a3w3_kd_gamma0_alpha10_sgd_lr0.004_wd1e-4_initPretrain
# # ========================================================================================


# # ====================================== FP, W4A4, W8A8 =============================================
# # ===== fp
# # if [ $METHOD_TYPE == "mobilenet_v2_fp" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --not-quant \
# #     --epochs=100


# # # ===== Pact, W4A4
# # elif [ $METHOD_TYPE == "teacher_lr0.004_wd1e-4/pact_overall_teacher58.07/a4w4/mobilenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/teacher_lr0.004_wd1e-4/mobilenetV2_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Pact + KD, W4A4
# # elif [ $METHOD_TYPE == "teacher_lr0.004_wd1e-4/pact_overall_teacher58.07/a4w4/mobilenetV2_pact_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/teacher_lr0.004_wd1e-4/mobilenetV2_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/teacher_lr0.004_wd1e-4/mobilenetV2_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100


# # # ===== Pact, W8A8
# # elif [ $METHOD_TYPE == "pact_overall/a8w8/mobilenetV2_pact_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Pact + KD, W8A8
# # elif [ $METHOD_TYPE == "pact_overall/a8w8/mobilenetV2_pact_a8w8_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100

# # # ===== LSQ, W4A4
# # elif [ $METHOD_TYPE == "lsq_overall/a4w4/mobilenetV2_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== LSQ + KD, W4A4
# # elif [ $METHOD_TYPE == "lsq_overall/a4w4/mobilenetV2_lsq_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100


# # # ===== Dorefa, W4A4
# # elif [ $METHOD_TYPE == "dorefa_overall/a4w4/mobilenetV2_dorefa_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100

# # # ===== Dorefa + KD, W4A4
# # elif [ $METHOD_TYPE == "dorefa_overall/a4w4/mobilenetV2_dorefa_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100


# # # ===== Dorefa, W8A8
# # elif [ $METHOD_TYPE == "dorefa_overall/a8w8/mobilenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== Dorefa + KD, W8A8
# # elif [ $METHOD_TYPE == "dorefa_overall/a8w8/mobilenetV2_dorefa_a8w8_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100
# # fi


# # # ===== LSQ, W8A8
# # elif [ $METHOD_TYPE == "lsq_overall/a8w8/mobilenetV2_lsq_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100

# # # ===== LSQ + KD, W8A8
# # elif [ $METHOD_TYPE == "lsq_overall/a8w8/mobilenetV2_lsq_a8w8_kd_gamma0_alpha300_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 300.0 \
# #     --epochs=100
# # fi
# # # ========================================================================================


# # ====================================== W3A3 =============================================
# # ===== Pact, W3A3
# # if [ $METHOD_TYPE == "pact_overall/a3w3/mobilenetV2_pact_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=1


# # # ===== Pact + KD, W3A3
# # elif [ $METHOD_TYPE == "pact_overall/a3w3/mobilenetV2_pact_a3w3_kd_gamma0_alpha10_sgd_lr0.004_wd1e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 10.0 \
# #     --epochs=100


# # ===== LSQ, W3A3
# # elif [ $METHOD_TYPE == "lsq_overall/a3w3/mobilenetV2_lsq_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # # ===== LSQ + KD, W3A3
# # elif [ $METHOD_TYPE == "lsq_overall/a3w3/mobilenetV2_lsq_a3w3_kd_gamma0_alpha10_sgd_lr0.004_wd1e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 10.0 \
# #     --epochs=100

# # ===== Dorefa, W3A3
# # elif [ $METHOD_TYPE == "dorefa_overall/a3w3/mobilenetV2_dorefa_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain" ]
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --epochs=100


# # ===== Dorefa + KD, W3A3
# # elif [ $METHOD_TYPE == "dorefa_overall/a3w3/mobilenetV2_dorefa_a3w3_kd_gamma0_alpha100_sgd_lr0.004_wd1e-4_initPretrain" ] 
# # then
# #     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 32 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 0.004 \
# #     --weight_decay 1e-4 \
# #     --backward_method "org" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100
# # fi
# # ========================================================================================


# # # # ====================================== PACT, W4A4, different backwards (Out of memory) =============================================
# # # PACT (forward: PACT, backward: EWGS) + KD
# # # Change --backward_method from "org" to "EWGS"
# # # ======================================================================================================================
# # # ===== PACT (forward: PACT, backward: EWGS) + KD, W4A4
# # # if [ $METHOD_TYPE == "pact_overall/a4w4/mobilenetV2_f:pact_b:ewgs_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
# # if [ $METHOD_TYPE == "test" ] 
# # then
# #     # python3.9 -m torch.distributed.launch --nproc_per_node=4  main.py \
# #     CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
# #     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
# #     -a mobilenet_v2 \
# #     --batch-size 64 \
# #     --loss-scale 128.0 \
# #     --workers 10 \
# #     --optimizer_type 'SGD' \
# #     --lr 5e-4 \
# #     --weight_decay 5e-4 \
# #     --backward_method "EWGS" \
# #     /home/users/kzhao27/tiny-imagenet-200 \
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --distill True \
# #     --teacher_arch mobilenet_v2 \
# #     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# #     --gamma 0.0 \
# #     --alpha 100.0 \
# #     --epochs=100
# # fi
# # # ======================================================================================================================



# # ====================================== Dorefa, W8A8, Effect of initialization (Results are not good without initing from teacher) ================================================
# # Random initialization, Remove: 
# #     --load_pretrain \
# #     --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
# # ******************************************************************************************************************************

# # ===== Dorefa, W8A8
# if [ $METHOD_TYPE == "dorefa_effect_of_initialization/a8w8/mobilenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ]
# then
#     CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
#     -a mobilenet_v2 \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --epochs=100


# # ===== Dorefa + KD, W8A8
# elif [ $METHOD_TYPE == "dorefa_effect_of_initialization/a8w8/mobilenetV2_dorefa_a8w8_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_withoutInitPretrain" ] 
# then
#     CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
#     --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
#     -a mobilenet_v2 \
#     --batch-size 32 \
#     --loss-scale 128.0 \
#     --workers 10 \
#     --optimizer_type 'SGD' \
#     --lr 5e-4 \
#     --weight_decay 5e-4 \
#     --backward_method "org" \
#     /home/users/kzhao27/tiny-imagenet-200 \
#     --distill True \
#     --teacher_arch mobilenet_v2 \
#     --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
#     --gamma 0.0 \
#     --alpha 100.0 \
#     --epochs=100
# fi
# # # ======================================================================================================================


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