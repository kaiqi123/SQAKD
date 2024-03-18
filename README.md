# Self-Supervised Quantization-Aware Knowledge Distillation (SQAKD)

This repo implements the paper published in **AISTATS 2024**:

**Self-Supervised Quantization-Aware Knowledge Distillation** (termed SQAKD)


## Installation
We implemented SQAKD on PyTorch version 1.10.0 and Python version 3.9.7.

We used four Nvidia RTX 2080 GPUs for model training and conduct inference experiments on Jetson Nano using NVIDIA TensorRT.

To get results on CIFAR-10/100, please follow the instructions in `CIFAR/scripts/requirement.sh`.

To get results on Tiny-Imagenet, please follow the instructions in `TinyImageNet/scripts/requirement.sh`.

## Running

1. For the results of "Improvements on SOTA QAT Methods":

    (1) For CIFAR-10 and CIFAR-100 results in Table 4 of the paper, follow the commands in `CIFAR/scripts/run_cifar10_resnet20.sh`, `CIFAR/scripts/run_cifar10_vgg8.sh`, `CIFAR/scripts/run_cifar100_resnet32.sh`, and `CIFAR/scripts/run_cifar100_vgg13.sh`.

    (2) For Tiny-ImageNet results of ResNet-18 and VGG-11 in Table 5 of the paper, follow the commands in `TinyImageNet/scripts/run_tinyimagenet_resnet18.sh` and `TinyImageNet/scripts/run_tinyimagenet_vgg11.sh`, respectively. 

    (2) For Tiny-ImageNet results of MobileNet-V2, ShuffleNet-V2, and SqueezeNet, in Table 6 of the paper, follow the commands in `TinyImageNet/scripts/run_tinyimagenet_mobilenetV2.sh`, `TinyImageNet/scripts/run_tinyimagenet_shufflenetv2.sh`, and `TinyImageNet/scripts/run_tinyimagenet_squeezenet.sh`, respectively. 

2. For the results of "Comparison with SOTA KD Methods" in Table 7 of the paper, follow the the commands in `CIFAR/scripts/run_cifar10_resnet20_EWGS+KDs.sh` and `CIFAR/scripts/run_cifar100_vgg13_EWGS+KDs.sh`. 


## Citation
If you think this repo is helpful for your research, please consider citing the paper: Will add paper reference after published




## Reference

@inproceedings{tian2019crd,
  title={Contrastive Representation Distillation},
  author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

@article{li2021mqbench,
  title={Mqbench: Towards reproducible and deployable model quantization benchmark},
  author={Li, Yuhang and Shen, Mingzhu and Ma, Jian and Ren, Yan and Zhao, Mingxin and Zhang, Qi and Gong, Ruihao and Yu, Fengwei and Yan, Junjie},
  journal={arXiv preprint arXiv:2111.03759},
  year={2021}
}

@inproceedings{lee2021network,
  title={Network Quantization with Element-wise Gradient Scaling},
  author={Lee, Junghyup and Kim, Dohyung and Ham, Bumsub},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}

## Reference License

The project of Mqbench is under Apache 2.0 License.

