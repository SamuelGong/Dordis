from abc import abstractmethod
from dordis.config import Config


class Handler:
    def __init__(self):
        self.total_clients = Config().clients.total_clients

    @abstractmethod
    def init_params(self, dim, q, target_num_clients):
        """ """

    @abstractmethod
    def encode_data(self, data, log_prefix_str, other_args):
        """ """

    @abstractmethod
    def decode_data(self, data, log_prefix_str, other_args):
        """ """

    @abstractmethod
    def get_bits(self):
        """ """

    @abstractmethod
    def get_padded_dim(self, dim):
        """ """
