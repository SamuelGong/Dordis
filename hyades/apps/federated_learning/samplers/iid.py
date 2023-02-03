import numpy as np
import torch
from hyades.config import Config
from torch.utils.data import SubsetRandomSampler

from hyades.apps.federated_learning.samplers import base


class Sampler(base.Sampler):
    def __init__(self, datasource, client_id):
        super().__init__(client_id)
        dataset = datasource.get_train_set()
        self.dataset_size = len(dataset)
        indices = list(range(self.dataset_size))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        partition_size = Config().app.data.partition_size
        total_clients = Config().clients.total_clients
        total_size = partition_size * total_clients

        # add extra samples to make it evenly divisible, if needed
        if len(indices) < total_size:
            while len(indices) < total_size:
                indices += indices[:(total_size - len(indices))]
        else:
            indices = indices[:total_size]
        assert len(indices) == total_size

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices[(int(client_id) -
                                       1):total_size:total_clients]

    def get(self):
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        version = torch.__version__
        if int(version[0]) <= 1 and int(version[2]) <= 5:
            return SubsetRandomSampler(self.subset_indices)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        return len(self.subset_indices)
