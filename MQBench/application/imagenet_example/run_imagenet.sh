# ================================== resnet-18
# 1. Server, train the model
# fp32
# python3.9 main.py -a resnet18 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories \
# --not-quant

# int8
# python3.9 main.py -a resnet18 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories


# 2. Server, deployment
# First export the quantized model to ONNX [tensorrt_deploy_model.onnx] and dump the clip ranges [tensorrt_clip_ranges.json] for activations.
# fp32
# python3.9 main.py -a resnet18 --resume results/resnet18_fp32/resnet18_fp32_model_best.pth.tar \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories \
# --evaluate \
# --not-quant

# # int8
# python3.9 main.py -a resnet18 --resume results/resnet18_int8/resnet18_int8_model_best.pth.tar --deploy \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories


# 3. Jetson Nano
# Second, go to jetson nano, build the TensorRT INT8 engine and evaluate, please make sure [dataset_path] contains subfolder [val].
# python3.9 onnx2trt.py \
# --onnx-path mqbench_qmodel_deploy_model.onnx \
# --trt-path mqbench_qmodel_deploy_resnet18.trt \
# --clip mqbench_qmodel_clip_ranges.json \
# --data /home/users/kzhao27/imagenet_data \
# --evaluate


# ======================================= MobileNet
# 1. Server, train the model
# fp32
# Acc@1 76.600 Acc@5 91.200
# python3.9 main.py -a mobilenet_v2 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories \
# --not-quant


# int 8
# Acc@1 56.600 Acc@5 82.200
# python3.9 main.py -a mobilenet_v2 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories


# 2. Server, Deployment
# First export the quantized model to ONNX [tensorrt_deploy_model.onnx] and dump the clip ranges [tensorrt_clip_ranges.json] for activations.
# fp32
# python3.9 main.py -a mobilenet_v2 --resume results/mobilenetV2_fp32/mobilenetV2_fp32_model_best.pth.tar \
# --train_data /home/users/kzhao27/imagenet_data_10categories \
# --val_data /home/users/kzhao27/imagenet_data_10categories \
# --evaluate \
# --not-quant

# int8
# python3.9 main.py -a mobilenet_v2 --resume model_best.pth.tar --deploy \
# --train_data /home/users/kzhao27/imagenet_data \
# --val_data /home/users/kzhao27/imagenet_data


# 3. Jetson Nano
# Second build the TensorRT INT8 engine and evaluate, please make sure [dataset_path] contains subfolder [val].
# CUDA_VISIBLE_DEVICES=0 
# python3.9 onnx2trt.py \
# --onnx-path mqbench_qmodel_deploy_model.onnx \
# --trt-path mqbench_qmodel_deploy_mobilenetV2.trt \
# --clip mqbench_qmodel_clip_ranges.json \
# --data /home/users/kzhao27/imagenet_data \
# --evaluate


# python3.9 onnx2trt.py \
# --trt-path mqbench_qmodel_deploy_mobilenetV2.trt \
# --data /home/users/kzhao27/imagenet_data \
# --evaluate