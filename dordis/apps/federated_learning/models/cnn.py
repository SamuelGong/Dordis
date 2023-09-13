import torch.nn as nn
import torch.nn.functional as F
from torch import manual_seed
import numpy as np

DDGAUSS = "ddgauss"  # means FEMNIST/MNIST
CIFAR10 = "cifar10"



class Model(nn.Module):
    def __init__(self, type, num_classes=10, seed=0):
        super().__init__()
        self.type = type
        manual_seed(3)

        if self.type == DDGAUSS:
            self.conv1 = nn.Conv2d(in_channels=1,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   bias=True)
        elif self.type == CIFAR10:
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   bias=True)
        # CIFAR10: 32, 32, 3 -> 30, 30, 32
        # FEMNIST: 28, 28, 1 -> 26, 26, 32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # CIFAR10: 30, 30, 32 -> 15, 15, 32
        # FEMNIST: 26, 26, 32 -> 13, 13, 32

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               bias=True)
        # CIFAR10: 15, 15, 32 -> 13, 13, 64
        # FEMNIST: 13, 13, 32 -> 11, 11, 64
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.flatten2 = nn.Flatten()
        # CIFAR10: 13*13*64 = 10816
        # FEMNIST: 11*11*64 = 7744

        if self.type == DDGAUSS:
            self.fc3 = nn.Linear(7744, 128)
        elif self.type == CIFAR10:
            self.fc3 = nn.Linear(10816, 128)

        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(128, num_classes)

        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc4.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.flatten2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

    @staticmethod
    def is_valid_model_type(model_type):
        return (model_type.startswith('cnn_')
                and len(model_type.split('_')) == 2
                and model_type.split('_')[1] in [DDGAUSS, CIFAR10])

    @staticmethod
    def get_model(model_type, num_classes=10):
        if not Model.is_valid_model_type(model_type):
            raise ValueError(
                'Invalid Resnet model type: {}'.format(model_type))

        cnn_type = model_type.split('_')[1]
        return Model(cnn_type, num_classes)


if __name__ == "__main__":
    model = Model.get_model("cnn_ddgauss")
    weights = model.cpu().state_dict()
    weight_info = {}

    weight_info['ordered_keys'] = list(weights.keys())
    weight_info['layer_shape'] = {}
    weight_info['trainable'] = []
    weight_info['non_trainable'] = []
    for k in weights.keys():
        shape = list(weights[k].size())
        if len(shape) > 0:
            weight_info['trainable'].append(k)
            weight_info['layer_shape'][k] = shape
        else:
            weight_info['non_trainable'].append(k)

    total_size = sum([int(np.prod(weight_info['layer_shape'][k]))
                      for k in weight_info['trainable']])

    print(f"{weight_info} {total_size} {total_size < 2 ** 20}.")
