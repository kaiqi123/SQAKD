# Self-Supervised Quantization-Aware Knowledge Distillation (SQAKD)

This repo implements the paper published in **AISTATS 2024**:

K. Zhao, M. Zhao, **Self-Supervised Quantization-Aware Knowledge Distillation,** Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS), May 2024. (Acceptance Rate: 27.6%)

The link of the paper is: https://arxiv.org/abs/2403.11106.

The overall workflow of SQAKD is as follows:

![Workflow of SQAKD](https://github.com/kaiqi123/SQAKD/Images/SQAKD_workflow.png)


## Installation
We implemented SQAKD on PyTorch version 1.10.0 and Python version 3.9.7.

We used four Nvidia RTX 2080 GPUs for model training and conduct inference experiments on Jetson Nano using NVIDIA TensorRT.

To run experiments on CIFAR-10/100, please follow the instructions in `CIFAR/scripts/requirement.sh`.

To run experiments on Tiny-Imagenet, please follow the instructions in `TinyImageNet/scripts/requirement.sh`.

## Running

1. For the results of "Improvements on SOTA QAT Methods":

    (1) For CIFAR-10 and CIFAR-100 results in Table 4 of the paper, follow the commands in `CIFAR/scripts/run_cifar10_resnet20.sh`, `CIFAR/scripts/run_cifar10_vgg8.sh`, `CIFAR/scripts/run_cifar100_resnet32.sh`, and `CIFAR/scripts/run_cifar100_vgg13.sh`.

    (2) For Tiny-ImageNet results of ResNet-18 and VGG-11 in Table 5 of the paper, follow the commands in `TinyImageNet/scripts/run_tinyimagenet_resnet18.sh` and `TinyImageNet/scripts/run_tinyimagenet_vgg11.sh`, respectively. 

    (2) For Tiny-ImageNet results of MobileNet-V2, ShuffleNet-V2, and SqueezeNet, in Table 6 of the paper, follow the commands in `TinyImageNet/scripts/run_tinyimagenet_mobilenetV2.sh`, `TinyImageNet/scripts/run_tinyimagenet_shufflenetv2.sh`, and `TinyImageNet/scripts/run_tinyimagenet_squeezenet.sh`, respectively. 

2. For the results of "Comparison with SOTA KD Methods" in Table 7 of the paper, follow the the commands in `CIFAR/scripts/run_cifar10_resnet20_EWGS+KDs.sh` and `CIFAR/scripts/run_cifar100_vgg13_EWGS+KDs.sh`. 


## Citation
If you think this repo is helpful for your research, please consider citing the paper:

```
@misc{zhao2024selfsupervised,
      title={Self-Supervised Quantization-Aware Knowledge Distillation}, 
      author={Kaiqi Zhao and Ming Zhao},
      year={2024},
      eprint={2403.11106},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Reference

Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive Representation Distillation." International Conference on Learning Representations. 2019.

Li, Yuhang, et al. "Mqbench: Towards reproducible and deployable model quantization benchmark." arXiv preprint arXiv:2111.03759 (2021).

Lee, Junghyup, Dohyung Kim, and Bumsub Ham. "Network quantization with element-wise gradient scaling." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

## Reference License

The project of Mqbench is under Apache 2.0 License.

