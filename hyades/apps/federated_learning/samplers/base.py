import os
from abc import abstractmethod

from hyades.config import Config


class Sampler:
    def __init__(self, client_id):
        if hasattr(Config().app.data, 'random_seed'):
            # so that every client can use different seed
            self.random_seed = Config().app.data.random_seed * client_id
        else:
            self.random_seed = os.getpid()

    @abstractmethod
    def get(self):
        """Obtains an instance of the sampler. """

    @abstractmethod
    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
