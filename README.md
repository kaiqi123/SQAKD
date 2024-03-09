# Self-Supervised Quantization-Aware Knowledge Distillation (SQAKD)

This repo implements the paper:

**Self-Supervised Quantization-Aware Knowledge Distillation** (termed SQAKD)


## Installation
We implemented SQAKD on PyTorch version 1.10.0 and Python version 3.9.7.

We used four Nvidia RTX 2080 GPUs for model training and conduct inference experiments on Jetson Nano using NVIDIA TensorRT.


## Running

1. For the results of "Improvements on SOTA QAT Methods":

    (1) For CIFAR-10 and CIFAR-100 results in Table 4 of the paper, follow the commands in the folder `CIFAR/scripts`. 

    (2) For Tiny-ImageNet results in Tables 5 and 6 of the paper, follow the commands in the folder `TinyImageNet/scripts`. 

2. For the results of "Comparison with SOTA KD Methods", follow the the commands in `CIFAR/scripts/run_cifar10_resnet20_EWGS+KDs.sh` and `CIFAR/scripts/run_cifar100_vgg13_EWGS+KDs.sh`. 



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
