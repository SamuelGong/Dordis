from abc import abstractmethod


class Handler:
    def __init__(self):
        pass

    @abstractmethod
    def set_seed(self, seed):
        """ """

    @abstractmethod
    def generate_numbers(self, num_range, dim):
        """ """