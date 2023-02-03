import time
import logging
import numpy as np
from hyades.config import Config
from hyades.client_samplers import base
from hyades.utils.share_memory_handler \
    import SAMPLED_CLIENTS, TRACE_RELATED
from hyades.utils.trace_related import read_trace, \
    raw_data_subsampling, sort_after_selection_dropout, \
    find_available, find_surviving


class ClientSampler(base.ClientSampler):
    def __init__(self, log_prefix_str):
        super(ClientSampler, self).__init__(log_prefix_str)
        self.is_simulation = hasattr(Config(), "simulation")
        self.seed = Config().clients.sample.seed
        np.random.seed(self.seed)

        if hasattr(Config().clients.sample, "sampling_rate_upperbound"):
            self.mode = "sampling_rate_upperbound"
            self.sampling_rate_upperbound \
                = Config().clients.sample.sampling_rate_upperbound
            self.num_sampled_client_upperbound = int(np.floor(
                self.total_clients * self.sampling_rate_upperbound
            ))
            self.selection_minimum = Config().clients.sample.selection_minimum
            logging.info(f"[Trace] Uniformly at random sample a subset of "
                         f"all available clients without replacement, "
                         f"where the sampling rate for each client "
                         f"does not exceed {self.sampling_rate_upperbound}.")
        else:
            self.mode = "fixed_sample_size"
            self.sample_size = Config().clients.sample.sample_size
            self.num_sampled_client_upperbound = self.sample_size
            self.worst_online_frac = Config().clients.worst_online_frac
            self.num_worst_online_clients \
                = int(np.floor(self.total_clients * self.worst_online_frac))
            self.sampling_rate_upperbound = self.sample_size \
                                            / self.num_worst_online_clients
            self.selection_minimum = Config().clients.sample.selection_minimum
            logging.info(f"[Trace] Uniformly at random sample {self.sample_size} "
                         f"out of all available clients without replacement.")

        self.threshold_frac = Config().agg.threshold
        # otherwise, aggregation will broken, which we current do not deal with

        self.mode = Config().clients.sample.params.mode
        if self.mode == "ranking":
            self.num_rounds = Config().app.repeat
            self.profiled_total_duration = 4 * 86400  # 4 days
            self.severity = Config().clients.sample.params.severity
            self.profiled_offset = Config().clients.sample.params.profiled_offset
            self.profiled_aggregation_latency \
                = Config().clients.sample.params.profiled_aggregation_latency
            self.profiled_round_duration \
                = Config().clients.sample.params.profiled_round_duration
            self.trace = None
            self.wall_clock_start = None
            self.idle_time = 0
        else:
            raise NotImplementedError

    def preprocessing(self):
        start_time = time.perf_counter()
        logging.info(f"Start to preprocess trace.")
        if self.mode == "ranking":
            raw_data, _ = read_trace(
                total_duration=self.profiled_total_duration,
                offset=self.profiled_offset
            )
            raw_data_sorted = sort_after_selection_dropout(
                raw_data=raw_data,
                num_rounds=self.num_rounds,
                round_duration=self.profiled_round_duration,
                aggregation_latency=self.profiled_aggregation_latency
            )
            raw_data_subsampled = raw_data_subsampling(
                raw_data_sorted=raw_data_sorted,
                trace_subsampling_rank=self.severity,
                total_clients=self.total_clients
            )

            client_id = 1
            self.trace = {}
            for trace_client in sorted(list(raw_data_subsampled.keys())):
                self.trace[client_id] = raw_data_subsampled[trace_client]
                client_id += 1
            # logging.info(f"[Debug] {self.trace}.")

        duration = round(time.perf_counter() - start_time, 2)
        logging.info(f"[{duration}s] Trace preprocessed.")

    def get_num_sampled_clients_upperbound(self):
        return self.num_sampled_client_upperbound

    def get_sampling_rate_upperbound(self):
        return self.sampling_rate_upperbound

    def sample(self, candidates, round_idx, log_prefix_str):  # TODO: this does not account for attendance
        if self.wall_clock_start is None:
            self.preprocessing()  # lazy initialization
            logging.info(f"{log_prefix_str} Start to simulate the trace.")
            self.wall_clock_start = time.perf_counter()

        while True:  # because we may not necessarily have enough clients to sample
            if self.is_simulation:
                current_time = round_idx * self.profiled_round_duration + self.idle_time
            else:
                current_time = round(time.perf_counter() - self.wall_clock_start, 2)

            # related_period_dict: used for dropout checking
            trace_candidates, related_period_dict \
                = find_available(self.trace, current_time)
            actual_candidates = [e for e in candidates if e in trace_candidates]

            if self.mode == "fixed_sample_size":
                sample_size = self.sample_size
            else:
                sample_size \
                    = int(np.floor(self.sampling_rate_upperbound * len(actual_candidates)))
            # logging.info(f"{current_time}s {len(actual_candidates)} {sample_size}.")

            if len(actual_candidates) < sample_size:
                sampled_clients = actual_candidates
            else:
                sampled_clients = np.random.choice(actual_candidates,
                                                   sample_size,
                                                   replace=False)
                # for serialization in multiprocessing
                sampled_clients = sorted([e.item() for e in sampled_clients])

            survivals = []
            aggregation_minimum = int(np.ceil(len(sampled_clients) * self.threshold_frac))
            if len(sampled_clients) >= self.selection_minimum:
                related_period_dict = {
                    k: related_period_dict[k] for k in related_period_dict.keys()
                    if k in sampled_clients
                }

                if self.is_simulation:
                    # also wants to make sure after-sampling dropout
                    # does not result in broken aggregation, e.g., none survives
                    survivals = find_surviving(
                        related_period_dict=related_period_dict,
                        current_time=current_time + self.profiled_aggregation_latency
                    )
                    if len(survivals) >= aggregation_minimum:
                        break
                else:
                    break

            sleep_time = 10  # TODO: avoid-harding
            if self.is_simulation:
                logging.info(f"Did not sample enough clients for round {round_idx}: "
                             f"{len(sampled_clients)} (Target: {self.selection_minimum}), "
                             f"or survivals {len(survivals)} (Target: {aggregation_minimum}). "
                             f"Wait and try again in {sleep_time}s.")
                self.idle_time += sleep_time
            else:
                logging.info(f"Did not sample enough clients for round {round_idx}: "
                             f"{len(sampled_clients)} (Target: {self.selection_minimum}). "
                             f"Wait and try again in {sleep_time}s.")
                time.sleep(sleep_time)

        logging.info(f"{log_prefix_str} Simulating trace for round {round_idx} at {current_time}s. "
                     f"original candidates ({len(candidates)}): {candidates}, "
                     f"actual candidates ({len(actual_candidates)}): {actual_candidates}, "
                     f"planned survivals ({len(survivals)}): {survivals}.")
        self.set_a_shared_value(
            key=[SAMPLED_CLIENTS, round_idx],
            value=sampled_clients
        )

        if self.is_simulation:
            self.set_a_shared_value(
                key=[TRACE_RELATED, round_idx],
                value={
                    "survivals": survivals
                }
            )
