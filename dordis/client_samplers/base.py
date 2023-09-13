from abc import abstractmethod
import numpy as np
from dordis.config import Config
from dordis.utils.share_memory_handler \
    import ShareBase, SAMPLED_CLIENTS, TRACE_RELATED, NEIGHBORS_DICT


class ClientSampler(ShareBase):
    def __init__(self, client_id) -> None:
        ShareBase.__init__(self, client_id=client_id)
        self.total_clients = Config().clients.total_clients

    @abstractmethod
    def sample(self, candidates, round_idx, log_prefix_str):
        """ Sampling clients to participate in a round. """

    @abstractmethod
    def get_sampling_rate_upperbound(self):
        """ """

    @abstractmethod
    def get_num_sampled_clients_upperbound(self):
        """ """


class ClientSamplePlugin(ShareBase):
    def __init__(self, client_id):
        ShareBase.__init__(self, client_id=client_id)
        self.sampled_clients_cached = {}
        self.trace_related_cached = {}
        self.neighbors_dict_cached = {}

    def client_id_transform(self, client_id, round_idx=None,
                            sampled_clients=None, mode="to_logical"):
        # clients/base.py does not have this function
        # as they are assumed to have no access to the following redis value
        if hasattr(Config().clients, "resource_saving") \
                and Config().clients.resource_saving:
            if round_idx is not None:  # server mode
                sampled_clients = self.fast_get_sampled_clients(round_idx)
            else:  # client mode
                assert sampled_clients is not None

            if mode == "to_logical":
                return sampled_clients[client_id - 1]
            else:  # to_physical
                return sampled_clients.index(client_id) + 1
        else:
            return client_id

    def fast_get_sampled_clients(self, round_idx):
        # hopefully somehow reduce redis pressure
        if round_idx not in self.sampled_clients_cached:
            sampled_clients = self.get_a_shared_value(
                key=[SAMPLED_CLIENTS, round_idx]
            )
            self.sampled_clients_cached[round_idx] = sampled_clients
        else:
            sampled_clients = self.sampled_clients_cached[round_idx]

        return sampled_clients

    def fast_get_neighbors_dict(self, round_idx):
        # hopefully somehow reduce redis pressure
        if round_idx not in self.neighbors_dict_cached:
            neighbors_dict = self.get_a_shared_value(
                key=[NEIGHBORS_DICT, round_idx]
            )
            self.neighbors_dict_cached[round_idx] = neighbors_dict
        else:
            neighbors_dict = self.neighbors_dict_cached[round_idx]

        return neighbors_dict

    def get_num_sampled_clients(self, round_idx):
        return len(self.fast_get_sampled_clients(round_idx))

    def fast_get_trace_related(self, round_idx):
        if round_idx not in self.trace_related_cached:
            trace_related = self.get_a_shared_value(
                key=[TRACE_RELATED, round_idx]
            )
            self.trace_related_cached[round_idx] = trace_related
        else:
            trace_related = self.trace_related_cached[round_idx]

        return trace_related

    def get_trace_related(self, round_idx):
        return self.fast_get_trace_related(round_idx)
