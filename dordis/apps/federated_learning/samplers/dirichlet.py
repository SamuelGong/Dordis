import logging
import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from dordis.config import Config

from dordis.apps.federated_learning.samplers import base


class Sampler(base.Sampler):
    def __init__(self, datasource, client_id):
        super().__init__(client_id)
        self.datasource = datasource
        self.sample_weights = None
        self.client_id = client_id
        np.random.seed(self.random_seed * int(client_id))

        concentration = Config().app.data.concentration if hasattr(
            Config().app.data, 'concentration') else 1.0
        self.update_concentration(concentration)

        partition_size = Config().app.data.partition_size
        self.update_partition_size(partition_size)

    def get(self):
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        return WeightedRandomSampler(weights=self.sample_weights,
                                     num_samples=self.partition_size,
                                     replacement=False,
                                     generator=gen)

    def trainset_size(self):
        return self.partition_size

    def update_partition_size(self, partition_size):
        self.partition_size = partition_size

    def update_concentration(self, concentration):
        target_list = self.datasource.targets()
        class_list = self.datasource.classes()

        target_proportions = np.random.dirichlet(
            np.repeat(concentration, len(class_list)))

        logging.info("[Client #%d] [Dirichlet] Target proportions: %s.",
                     self.client_id, target_proportions)

        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[random.randint(0, len(class_list) - 1)] = 1

        self.sample_weights = target_proportions[target_list]
