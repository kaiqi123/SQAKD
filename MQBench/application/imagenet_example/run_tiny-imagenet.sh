# =====================================================================================
# argument -a/--arch: (choose from 
# 'alexnet', 
# 'densenet121', 'densenet161', 'densenet169', 'densenet201', 
# 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 
# 'googlenet', 
# 'inception_v3', 
# 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 
# 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 
# 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 
# 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', '
# shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 
# 'squeezenet1_0', 'squeezenet1_1', 
# 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
# 'wide_resnet101_2', 'wide_resnet50_2')
# =====================================================================================


# ************************************ tiny-imagenet-200, resnet-18

# 1. Server, train the model
# fp32
# python3.9 main.py -a resnet18 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/tiny-imagenet-200 \
# --val_data /home/users/kzhao27/tiny-imagenet-200 \
# --not-quant

# int8
# python3.9 main.py -a resnet18 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/tiny-imagenet-200 \
# --val_data /home/users/kzhao27/tiny-imagenet-200


# ************************************ tiny-imagenet-200, mobilenet_v2
# 1. Server, train the model
# int8
#  * Acc@1 3.790 Acc@5 12.240
# python3.9 main.py -a mobilenet_v2 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/tiny-imagenet-200 \
# --val_data /home/users/kzhao27/tiny-imagenet-200



# ************************************ tiny-imagenet-200, efficientnet_b0
# 1. Server, train the model
# int8
#  * Acc@1 0.210 Acc@5 0.870
# python3.9 main.py -a efficientnet_b0 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/tiny-imagenet-200 \
# --val_data /home/users/kzhao27/tiny-imagenet-200



# ************************************ tiny-imagenet-200, shufflenet_v2_x0_5
# 1. Server, train the model
# int8
#  * Acc@1 0.780 Acc@5 3.620
# python3.9 main.py -a shufflenet_v2_x0_5 --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/tiny-imagenet-200 \
# --val_data /home/users/kzhao27/tiny-imagenet-200


# ************************************ tiny-imagenet-200, googlenet
# 1. Server, train the model
# int8
#  * Acc@1 0.860 Acc@5 3.230
# python3.9 main.py -a googlenet --epochs 1 --lr 1e-4 --batch-size 128 --pretrained \
# --train_data /home/users/kzhao27/tiny-imagenet-200 \
# --val_data /home/users/kzhao27/tiny-imagenet-200