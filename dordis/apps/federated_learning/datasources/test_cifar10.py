import logging

from torchvision import datasets, transforms
import os
import time
# from dordis.config import Config
# from dordis.apps.federated_learning.datasources import base


if __name__ == "__main__":
    _path = '.'

    _transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = datasets.CIFAR10(root=_path,
                                     train=True,
                                     download=True,
                                     transform=_transform)
    testset = datasets.CIFAR10(root=_path,
                                    train=False,
                                    download=True,
                                    transform=_transform)

