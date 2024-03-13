

# METHOD_TYPE=$1
# echo $METHOD_TYPE



# === squeezenet1_0
# METHOD_TYPE=\
# 'results/tiny-imagenet_squeezenet/squeezenet_fp/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_squeezenet/dorefa_overall/a8w8/squeezenet_dorefa_a8w8_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_squeezenet/dorefa_overall/a8w8/squeezenet_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_squeezenet/lsq_overall/a4w4/squeezenet_lsq_a4w4_kd_gamma0_alpha50_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_squeezenet/lsq_overall/a4w4/squeezenet_lsq_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'

# CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# --save_root_path "results/evaluate_models_tiny-imagenet/$METHOD_TYPE" \
# -a squeezenet1_0 \
# --batch-size 64 \
# --backward_method "org" \
# /home/users/kzhao27/tiny-imagenet-200 \
# --resume "$METHOD_TYPE" \
# --evaluate \
# --not-quant \
# --epochs=100



# === shufflenet_v2
METHOD_TYPE=\
'results/tiny-imagenet_shufflenetV2/dorefa_overall/a8w8/shufflenetV2_dorefa_a8w8_kd_gamma0.0_alpha800.0_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# 'results/tiny-imagenet_shufflenetV2/dorefa_overall/a8w8/shufflenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# 'results/tiny-imagenet_shufflenetV2/pact_overall/a4w4/shufflenetV2_pact_a4w4_kd_gamma0.0_alpha100.0_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# 'results/tiny-imagenet_shufflenetV2/pact_overall/a4w4/shufflenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# 'results/tiny-imagenet_shufflenetV2/shufflenetV2_fp/checkpoints/model_best.pth.tar'

CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
--save_root_path "results/evaluate_models_tiny-imagenet/$METHOD_TYPE" \
-a shufflenet_v2_x0_5 \
--batch-size 64 \
--backward_method "org" \
/home/users/kzhao27/tiny-imagenet-200 \
--resume "$METHOD_TYPE" \
--evaluate \
--epochs=100
# --not-quant \




# === mobilenetV2
# METHOD_TYPE=\
# 'results/tiny-imagenet_mobilenetV2/mobilenetV2_fp/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_mobilenetV2/dorefa_overall/a8w8/mobilenetV2_dorefa_a8w8_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_mobilenetV2/dorefa_overall/a8w8/mobilenetV2_dorefa_a8w8_independent_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_mobilenetV2/pact_overall/a4w4/mobilenetV2_pact_a4w4_kd_gamma0_alpha100_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_mobilenetV2/pact_overall/a4w4/mobilenetV2_pact_a4w4_independent_sgd_lr5e-4_wd5e-4_initPretrain/checkpoints/model_best.pth.tar'
# # 'results/tiny-imagenet_mobilenetV2/pact_overall/a3w3/mobilenetV2_pact_a3w3_independent_sgd_lr0.004_wd1e-4_initPretrain/checkpoints/model_best.pth.tar' \
# # 'results/tiny-imagenet_mobilenetV2/pact_overall/a3w3/mobilenetV2_pact_a3w3_kd_gamma0_alpha10_sgd_lr0.004_wd1e-4_initPretrain/checkpoints/model_best.pth.tar' \

# CUDA_VISIBLE_DEVICES=0 python3.9 main.py \
# --save_root_path "results/evaluate_models_tiny-imagenet/$METHOD_TYPE" \
# -a mobilenet_v2 \
# --batch-size 64 \
# --backward_method "org" \
# /home/users/kzhao27/tiny-imagenet-200 \
# --resume "$METHOD_TYPE" \
# --evaluate \
# --not-quant \
# --epochs=100

