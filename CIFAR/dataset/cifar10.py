from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import sys

'''
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50,000 training images and 10,000 test images.

'''

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



class CIFAR10InstanceSample(datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0): 
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
                
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])
        

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) 
        self.cls_negative = np.asarray(self.cls_negative)
      
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False 
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) 
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

            return img, target, index, sample_idx



def get_cifar10_dataloaders(data_folder):
    train_dataset = datasets.CIFAR10(root=data_folder,
                            train=True, 
                            transform=transform_train,
                            download=True)
    
    test_dataset = datasets.CIFAR10(root=data_folder,
                            train=False, 
                            transform=transform_test)
    return train_dataset, test_dataset



def get_cifar10_dataloaders_sample(data_folder, k=4096, mode='exact', is_sample=True, percent=1.0):

    train_dataset = CIFAR10InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=transform_train,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)

    test_dataset = datasets.CIFAR10(root=data_folder,
                            train=False, 
                            transform=transform_test)

    return train_dataset, test_dataset
