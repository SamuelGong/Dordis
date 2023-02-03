import logging
import numpy as np
from hyades.config import Config
from hyades.client_samplers import base
from hyades.utils.share_memory_handler import SAMPLED_CLIENTS


class ClientSampler(base.ClientSampler):
    def __init__(self, log_prefix_str):
        super(ClientSampler, self).__init__(log_prefix_str)
        seed = Config().clients.sample.seed
        np.random.seed(seed)

        self.mode = Config().clients.sample.mode
        self.num_sampled_client_upperbound = None
        self.sampling_rate_upperbound = None

        if self.mode == "fixed_sample_size":
            assert hasattr(Config().clients.sample, "sample_size")
            self.sample_size = Config().clients.sample.sample_size
            self.worst_online_frac = Config().clients.worst_online_frac
            self.num_worst_online_clients \
                = int(np.floor(self.total_clients * self.worst_online_frac))

            self.num_sampled_client_upperbound = self.sample_size
            self.sampling_rate_upperbound = self.sample_size \
                                            / self.num_worst_online_clients

            logging.info(f"[Uniform] Uniformly at random sample {self.sample_size} "
                         f"out of all available clients without replacement.")
        elif self.mode == "sampling_rate_upperbounded":
            assert hasattr(Config().clients.sample, "sampling_rate_upperbound")

            self.sampling_rate_upperbound \
                = Config().clients.sample.sampling_rate_upperbound
            self.num_sampled_client_upperbound = int(np.floor(
                self.total_clients * self.sampling_rate_upperbound
            ))

            logging.info(f"[Uniform] Uniformly at random sample a subset of "
                         f"all available clients without replacement, "
                         f"where the sampling rate for each client "
                         f"does not exceed {self.sampling_rate_upperbound}.")

    def get_num_sampled_clients_upperbound(self):
        return self.num_sampled_client_upperbound

    def get_sampling_rate_upperbound(self):
        return self.sampling_rate_upperbound

    def sample(self, candidates, round_idx, log_prefix_str):
        if self.mode == "fixed_sample_size":
            sample_size = self.sample_size
        else:  # self.mode == "sampling_rate_upperbounded"
            sample_size \
                = int(np.floor(self.sampling_rate_upperbound * len(candidates)))

        if len(candidates) >= sample_size and len(candidates) > 0:
            sampled_clients = np.random.choice(candidates,
                                               sample_size,
                                               replace=False)
            # for serialization in multiprocessing
            sampled_clients = sorted([e.item() for e in sampled_clients])
            self.set_a_shared_value(
                key=[SAMPLED_CLIENTS, round_idx],
                value=sampled_clients
            )
        else:  # has to stop due to privacy and any other practical concerns
            logging.info(f"[Uniform] No clients are sampled "
                         f"due to insufficient candidates "
                         f"({len(candidates)}<{sample_size}).")
