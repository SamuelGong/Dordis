import logging
from dordis.client_samplers import base
from dordis.utils.share_memory_handler import SAMPLED_CLIENTS


class ClientSampler(base.ClientSampler):
    def __init__(self, log_prefix_str):
        super(ClientSampler, self).__init__(log_prefix_str)
        logging.info(f"Sample all available "
                     f"clients at each round.")

    def sample(self, candidates, round_idx, log_prefix_str):
        if len(candidates) > 0:
            self.set_a_shared_value(
                key=[SAMPLED_CLIENTS, round_idx],
                value=candidates
            )
        else:
            logging.info(f"[All inclusive] No clients are sampled "
                         f"due to no candidates.")

    def get_sampling_rate_upperbound(self):
        return 1.0

    def get_num_sampled_clients_upperbound(self):
        return self.total_clients
