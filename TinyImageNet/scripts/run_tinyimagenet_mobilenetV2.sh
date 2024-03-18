# conda activate mqbench
#!/bin/bash


# ====================== Tiny-ImageNet, MobileNet-V2 ===============================================================================
# SGD
# batch-size 
#     PACT: 64
#     LSQ, DoReFa: 32
# lr, weight_decay, initilization
#     fp: lr: 0.004, weight_decay: 1e-4
#     Quantized: 5e-4, weight_decay: 5e-4, initilized with pretrained fp model

# Note that:
# 1. You need to choose function of "get_extra_config()" in get_config.py, based on the QAT method, including PACT, LSQ, and DoReFa
# 2. Modify the bit-widths for both weights and activations, inside the function of "get_extra_config()",
#    E.g., the setting of 4-bit is as follows: 'bit': 4
# ==================================================================================================================================


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE


# ===== fp
if [ $METHOD_TYPE == "mobilenet_v2_fp" ] 
then
    CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
    -a mobilenet_v2 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 0.004 \
    --weight_decay 1e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --not-quant \
    --epochs=100

# ************************************** W4A4
# ===== W4A4, Pact
elif [ $METHOD_TYPE == "teacher_lr0.004_wd1e-4/pact_overall_teacher58.07/a4w4/mobilenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
then
    CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
    -a mobilenet_v2 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_mobilenetV2/teacher_lr0.004_wd1e-4/mobilenetV2_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
    --epochs=100


# ===== W4A4, Pact + KD
elif [ $METHOD_TYPE == "teacher_lr0.004_wd1e-4/pact_overall_teacher58.07/a4w4/mobilenetV2_pact_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain" ] 
then
    CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
    -a mobilenet_v2 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_mobilenetV2/teacher_lr0.004_wd1e-4/mobilenetV2_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
    --distill True \
    --teacher_arch mobilenet_v2 \
    --teacher_path "./results/tiny-imagenet_mobilenetV2/teacher_lr0.004_wd1e-4/mobilenetV2_fp_lr0.004_wd1e-4/checkpoints/model_best.pth.tar" \
    --gamma 0.0 \
    --alpha 100.0 \
    --epochs=100


# ************************************** W3A3
# ===== W3A3, Pact
if [ $METHOD_TYPE == "pact_overall/a3w3/mobilenetV2_pact_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain" ]
then
    CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
    -a mobilenet_v2 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 0.004 \
    --weight_decay 1e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
    --epochs=1


# =====  W3A3, Pact + KD
elif [ $METHOD_TYPE == "pact_overall/a3w3/mobilenetV2_pact_a3w3_kd_gamma0_alpha10_sgd_lr0.004_wd1e-4_initPretrain" ] 
then
    CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_mobilenetV2/$METHOD_TYPE" \
    -a mobilenet_v2 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 0.004 \
    --weight_decay 1e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
    --distill True \
    --teacher_arch mobilenet_v2 \
    --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar" \
    --gamma 0.0 \
    --alpha 10.0 \
    --epochs=100
fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"