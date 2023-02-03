import redis
import logging
import numpy as np
from hyades.protocols import secagg
from hyades.config import Config
from hyades.primitives.differential_privacy \
    import registry as dp_registry
from hyades.primitives.differential_privacy\
    .utils.misc import modular_clip
from hyades.utils.debug import log_sketch
from hyades.utils.quantizer import quantize
from hyades.utils.batcher import batch
from hyades.utils.share_memory_handler \
    import redis_pool


r = redis.Redis(connection_pool=redis_pool)


class ProtocolServer(secagg.ProtocolServer):
    def __init__(self, client_id=0):
        super().__init__(client_id=0)
        self.init_sample_hadamard_seed = Config().agg \
            .differential_privacy.params.init_sample_hadamard_seed

        self.dp_handler = dp_registry.get()

    def padding_dim(self, dim):
        if Config().agg.differential_privacy.mechanism \
                in ["dskellam", "ddgauss"]:
            return int(np.math.pow(2, np.ceil(np.log2(dim))))
        else:
            return dim

    def post_calc_chunk_size(self, chunk_size, calc_dp_params=True):
        if calc_dp_params:
            for chunk_idx, chunk_dim in chunk_size.items():
                dp_params_dict = self.get_a_shared_value(
                    key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
                )
                if dp_params_dict is None:  # avoid redundant computation
                    dp_params_dict = self.dp_handler.init_params(
                        dim=chunk_dim,
                        q=1.0,  # there is no amplification via subsampling
                        target_num_clients=self.num_sampled_clients_upperbound
                    )
                    self.set_a_shared_value(
                        key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx],
                        value=dp_params_dict
                    )

    def encode_data(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.ENCODE_DATA
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        prepared_data = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.PREPARE_DATA
        )
        surviving_clients = sorted(list(prepared_data.keys()))

        customized_idx = Config().app.repeat * round_idx + chunk_idx
        sample_hadamard_seed = self.init_sample_hadamard_seed * customized_idx
        logging.info("%s Phase started. Instructing %d "
                     "clients to encode data using "
                     "sample hamazard seed %d: %s.", log_prefix_str,
                     len(surviving_clients), sample_hadamard_seed,
                     surviving_clients)
        self.set_a_shared_value(
            key=["sample_hadamard_seed", round_idx, chunk_idx],
            value=sample_hadamard_seed
        )

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'sample_hadamard_seed': sample_hadamard_seed
            },
            "send_list": surviving_clients,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }


class ProtocolClient(secagg.ProtocolClient):
    def __init__(self, client_id):
        super().__init__(client_id)
        self.dp_handler = dp_registry.get()

    def padding_dim(self, dim):
        if Config().agg.differential_privacy.mechanism \
                in ["dskellam", "ddgauss"]:
            return int(np.math.pow(2, np.ceil(np.log2(dim))))
        else:
            return dim

    def post_calc_chunk_size(self, chunk_size, calc_dp_params=True):
        if calc_dp_params:
            for chunk_idx, chunk_dim in chunk_size.items():
                dp_params_dict = self.get_a_shared_value(
                    key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
                )
                if dp_params_dict is None:  # avoid redundant computation
                    dp_params_dict = self.dp_handler.init_params(
                        dim=chunk_dim,
                        q=1.0,  # there is no amplification via subsampling
                        target_num_clients=self.num_sampled_clients_upperbound
                    )
                    self.set_a_shared_value(
                        key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx],
                        value=dp_params_dict
                    )

    def encode_data(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        sample_hadamard_seed = payload["sample_hadamard_seed"]
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        self.set_a_shared_value(
            key=['sample_hadamard_seed', round_idx, chunk_idx],
            value=sample_hadamard_seed
        )
        data = self.get_a_shared_value(
            key=['data', round_idx, chunk_idx]
        )
        full_data_norm = self.get_a_shared_value(
            key=["full_data_norm", round_idx]
        )

        # need busy waiting as at the time of calling this method
        # calc_chunk_size() may have not been called
        # especially with high concurrency
        dp_params_dict = self.get_a_shared_value(
            key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx],
            busy_waiting=True,
        )

        num_sampled_clients \
            = self.get_num_sampled_clients(round_idx=round_idx)

        if hasattr(Config().agg, "differential_privacy"):
            if hasattr(Config().agg.differential_privacy,
                    "dropout_resilience"):
                xnoise_params = self.dp_handler.get_xnoise_params(
                    round_idx=round_idx,
                    num_sampled_clients=num_sampled_clients,
                    dp_params_dict=dp_params_dict
                )
            else:
                xnoise_params = None

            data, excessive_noise_seeds = self.dp_handler.encode_data(
                data=data,
                log_prefix_str=log_prefix_str,
                other_args=(sample_hadamard_seed, num_sampled_clients,
                            dp_params_dict, full_data_norm, xnoise_params)
            )

            if excessive_noise_seeds is not None:
                self.set_a_shared_value(
                    key=["excessive_noise_seeds", round_idx, chunk_idx],
                    value=excessive_noise_seeds
                )

        log_sketch(data, log_prefix_str, mode="client")

        # as we are using SecAgg, even if we do not batch
        # we still need to make it positive
        bits_per_element = self.dp_handler.get_bits()
        data = (np.array(data) + 2 ** (bits_per_element - 1)).tolist()
        if hasattr(Config().agg.differential_privacy, "batch"):
            data = self.batch_data(
                data=data,
                batching_params=Config().agg.differential_privacy,
                bits_per_element=bits_per_element,
                log_prefix_str=log_prefix_str
            )

        self.set_a_shared_value(
            key=['data', round_idx, chunk_idx],
            value=data
        )

        return {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Data encoded."
        }
