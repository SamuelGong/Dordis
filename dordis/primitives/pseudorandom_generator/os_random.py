import random
from dordis.primitives.pseudorandom_generator import base


class Handler(base.Handler):
    def __init__(self):
        super().__init__()
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)

    def generate_numbers(self, num_range, dim):
        res = [random.randrange(start=num_range[0], stop=num_range[1]) for _ in range(dim)]
        return res
