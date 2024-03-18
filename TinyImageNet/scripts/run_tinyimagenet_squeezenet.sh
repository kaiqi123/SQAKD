# conda activate mqbench
#!/bin/bash


# ====================== Tiny-ImageNet, SqueezeNet ===============================================================================
# SGD
# batch-size: 64
# lr, weight_decay, initilization
#     fp: lr: 5e-2, weight_decay: 5e-4, acc: 51.49
#     4-bit and 8-bit Quantization: 5e-4, weight_decay: 5e-4, initilized with pretrained fp model

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
if [ $METHOD_TYPE == "squeezenet_fp" ] 
then
    CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
    -a squeezenet1_0 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-2 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --not-quant \
    --epochs=100


# ===== Dorefa, W8A8
elif [ $METHOD_TYPE == "dorefa_overall/a8w8/squeezenet_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
then
    CUDA_VISIBLE_DEVICES=2 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
    -a squeezenet1_0 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
    --epochs=100


# ===== Dorefa + KD, W8A8
elif [ $METHOD_TYPE == "dorefa_overall/a8w8/squeezenet_dorefa_a8w8_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain" ] 
then
    CUDA_VISIBLE_DEVICES=3 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
    -a squeezenet1_0 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
    --distill True \
    --teacher_arch squeezenet1_0 \
    --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
    --gamma 0.0 \
    --alpha 50.0 \
    --epochs=100


# ===== LSQ, W4A4
elif [ $METHOD_TYPE == "lsq_overall/a4w4/squeezenet_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain" ]
then
    CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
    -a squeezenet1_0 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
    --epochs=100


# ===== LSQ + KD, W4A4
elif [ $METHOD_TYPE == "lsq_overall/a4w4/squeezenet_lsq_a4w4_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain" ] 
then
    CUDA_VISIBLE_DEVICES=1 python3.9 main.py \
    --save_root_path "results/tiny-imagenet_squeezenet/$METHOD_TYPE" \
    -a squeezenet1_0 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers 10 \
    --optimizer_type 'SGD' \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
    --distill True \
    --teacher_arch squeezenet1_0 \
    --teacher_path "./results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar" \
    --gamma 0.0 \
    --alpha 50.0 \
    --epochs=100
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"