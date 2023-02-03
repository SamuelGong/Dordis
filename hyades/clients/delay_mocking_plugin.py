import logging
import numpy as np
from hyades.config import Config
from hyades.utils.misc import my_random_zipfian
from hyades.utils.share_memory_handler import ShareBase, DELAY_MOCKING_FACTORS


class DelayMockingPlugin(ShareBase):
    def __init__(self, client_id):
        ShareBase.__init__(self, client_id=client_id)
        self.delay_mocking_factors_cached = None

    def calculate_delay_mocking_factors(self):
        total_clients = Config().clients.total_clients
        delay_mocking_factors = np.zeros(total_clients).tolist()

        type = Config().clients.delay_mock.type
        if type == "proportional":
            args = Config().clients.delay_mock.args
            dist = args[0]
            if dist == "zipf":
                seed = Config().clients.delay_mock.seed

                a, amin, amax, shuffle = args[1:5]
                np.random.seed(seed)
                factors = my_random_zipfian(
                    a=a,
                    n=total_clients,
                    amin=amin,
                    amax=amax
                )
                delay_mocking_factors = sorted(factors)
                if shuffle:
                    # so that it can be independent of
                    # the communication bandwidth (which is sorted)
                    np.random.shuffle(delay_mocking_factors)
            elif dist == "constant":
                c = args[1]
                delay_mocking_factors = [c] * total_clients

        self.set_a_shared_value(
            key=DELAY_MOCKING_FACTORS,
            value=delay_mocking_factors
        )
        logging.info(f"Delay mocking factors calculated: {delay_mocking_factors}. "
                     f"Stored at DB.")

    def fast_get_delay_mocking_factors(self, logical_client_id):
        if self.delay_mocking_factors_cached is not None:
            delay_mocking_factors = self.delay_mocking_factors_cached
        else:
            delay_mocking_factors = self.get_a_shared_value(
                key=DELAY_MOCKING_FACTORS
            )
            if delay_mocking_factors is None:
                self.calculate_delay_mocking_factors()
                delay_mocking_factors = self.get_a_shared_value(
                    key=DELAY_MOCKING_FACTORS
                )
            self.delay_mocking_factors_cached = delay_mocking_factors

        delay_mocking_factor = delay_mocking_factors[logical_client_id - 1]
        logging.info(f"Got delay mocking factor for "
                     f"logical client {logical_client_id}: "
                     f"{delay_mocking_factor}.")
        return delay_mocking_factor
