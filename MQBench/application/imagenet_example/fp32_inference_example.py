import torchvision.models as models
import torch
import torch.onnx


# ============================ 
# Refer to: https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
# ============================

# 1. What format should I save my model in?

# resnet50
# # load the pretrained model
# resnet50 = models.resnet50(pretrained=True, progress=False).eval()

# BATCH_SIZE=32
# dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)

# # export the model to ONNX
# torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)


# ============== ResNet-18
# # load the pretrained model
# resnet18 = models.resnet18(pretrained=True, progress=False).eval()

# BATCH_SIZE=32
# dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)

# # export the model to ONNX
# torch.onnx.export(resnet18, dummy_input, "input_models/tinyImagenet_resnet18_fp32/tinyImagenet_resnet18_pytorch.onnx", verbose=False)




# ============== VGG-11
# load the pretrained model
model = models.vgg11_bn(pretrained=True, progress=False).eval()

BATCH_SIZE=32
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)

# export the model to ONNX
torch.onnx.export(model, dummy_input, "results/vgg11_pytorch.onnx", verbose=False)

