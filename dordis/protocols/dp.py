import gc
import redis
import logging
import numpy as np
import multiprocessing as mp
from dordis.protocols import plaintext
from dordis.config import Config
from dordis.primitives.differential_privacy \
    import registry as dp_registry
from dordis.primitives.differential_privacy\
    .utils.misc import modular_clip
from dordis.utils.share_memory_handler \
    import redis_pool, SCHEDULE
from dordis.utils.debug import log_sketch
from dordis.utils.misc import plaintext_aggregate, plaintext_add

N_CPUS = mp.cpu_count()
r = redis.Redis(connection_pool=redis_pool)


class ProtocolServer(plaintext.ProtocolServer):
    def __init__(self, client_id=0):
        super().__init__(client_id=client_id)
        if hasattr(Config().agg.differential_privacy.params,
                   "init_sample_hadamard_seed"):  # only used for ddgauss, dskellam
            self.init_sample_hadamard_seed = Config().agg\
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
        return_dict = {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx
            },
            "send_list": surviving_clients,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }
        if hasattr(self, "init_sample_hadamard_seed"):
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
            return_dict["payload"]["sample_hadamard_seed"] = sample_hadamard_seed

        return return_dict

    def generate_output(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.GENERATE_OUTPUT
        client_data_dict = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.UPLOAD_DATA
        )
        dp_params_dict = self.get_a_shared_value(
            key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
        )

        surviving_clients = list(client_data_dict.keys())
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        self.update_stat(
            clients=surviving_clients,
            log_prefix_str=log_prefix_str,
            round_idx=round_idx,
            chunk_idx=chunk_idx
        )
        logging.info("%s Phase started. Generating output "
                     "with %d clients: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        if hasattr(Config().agg.differential_privacy, "batch"):
            bits_per_element = self.dp_handler.get_bits()
            original_length = self.dp_handler.get_padded_dim(
                dim=self.chunk_size[chunk_idx]
            )
            for k, v in client_data_dict.items():
                # logging.info(f"[Debug] {k}, {len(v)}")
                unbatched_v = self.unbatch_data(
                    data=v,
                    batching_params=Config().agg.differential_privacy,
                    bits_per_element=bits_per_element,
                    original_length=original_length,
                    log_prefix_str=log_prefix_str
                )
                unbatched_v = (np.array(unbatched_v) - 2 ** (bits_per_element - 1)).tolist()
                client_data_dict[k] = unbatched_v

        agg_res = plaintext_aggregate(
            data_list=list(client_data_dict.values())
        )

        agg_res = modular_clip(
            data=agg_res,
            log_prefix_str=log_prefix_str,
            other_args=(dp_params_dict,)
        )

        self.batch_set_shared_values(d={
            "agg_res": agg_res,
            "involved_clients": surviving_clients,
        },
            postfix=[round_idx, chunk_idx],)

        self._publish_a_value(
            channel=SCHEDULE,
            message=[round_idx, chunk_idx, phase_idx]
        )
        logging.info("%s Phase done.", log_prefix_str)
        return {}

    # def unquantize_data(self, data, quantization_params, log_prefix_str,
    #                     for_addition=True, num_involve_clients=None,
    #                     padded_num_bits_to_subtract=None, aux=None):
    #     # for_addition, num_involve_clients, and padded_num_bits_to_subtract
    #     # are not relevant here; they are used for basic unquantization
    #     assert aux is not None
    #
    #     logging.info(f"[Debug] Before unquantization: "
    #                  f"{[round(e, 4) for e in data[:3]]} "
    #                  f"{[round(e, 4) for e in data[-3:]]}.")
    #
    #     round_idx, chunk_idx = aux
    #     dp_params_dict = self.get_a_shared_value(
    #         key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
    #     )
    #     sample_hadamard_seed = self.get_a_shared_value(
    #         key=["sample_hadamard_seed", round_idx, chunk_idx]
    #     )
    #     data = self.dp_handler.decode_data(
    #         data=data,
    #         log_prefix_str=log_prefix_str,
    #         other_args=(sample_hadamard_seed, dp_params_dict)
    #     )
    #
    #     logging.info(f"[Debug] After unquantization: "
    #                  f"{[round(e, 4) for e in data[:3]]} "
    #                  f"{[round(e, 4) for e in data[-3:]]}.")
    #     logging.info("%s Unquantization done.", log_prefix_str)
    #     return data

    def clean_a_chunk(self, round_idx, chunk_idx):
        self.delete_records_for_phases(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phases=[self.PREPARE_DATA, self.UPLOAD_DATA,
                    self.CLIENT_USE_OUTPUT]
        )
        self.batch_delete_shared_values(
            keys=['agg_res', 'sample_hadamard_seed'],
            postfix=[round_idx, chunk_idx]
        )
        self.clients_dropped_out = []
        gc.collect()


class ProtocolClient(plaintext.ProtocolClient):
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
                        # q=self.client_sampling_rate_upperbound,
                        q=1.0,  # there is no amplification via subsampling
                        target_num_clients=self.num_sampled_clients_upperbound
                    )
                    self.set_a_shared_value(
                        key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx],
                        value=dp_params_dict
                    )

    def encode_data(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )

        if "sample_hadamard_seed" in payload:
            sample_hadamard_seed = payload["sample_hadamard_seed"]
            self.set_a_shared_value(
                key=['sample_hadamard_seed', round_idx, chunk_idx],
                value=sample_hadamard_seed
            )
        else:
            sample_hadamard_seed = None
        data = self.get_a_shared_value(
            key=['data', round_idx, chunk_idx]
        )
        dp_params_dict = self.get_a_shared_value(
            key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
        )
        full_data_norm = self.get_a_shared_value(
            key=["full_data_norm", round_idx]
        )

        num_sampled_clients = self.get_num_sampled_clients(round_idx=round_idx)
        data, _ = self.dp_handler.encode_data(
            data=data,
            log_prefix_str=log_prefix_str,
            other_args=(sample_hadamard_seed, num_sampled_clients,
                        dp_params_dict, full_data_norm, None)
            # the None is for xnoise_params,
            # both of which are not enabled in dp or with DDGauss
            # but in dp_plus_secagg with DSkallem
        )
        log_sketch(data, log_prefix_str, mode="client")

        if hasattr(Config().agg.differential_privacy, "batch"):
            bits_per_element = self.dp_handler.get_bits()
            data = (np.array(data) + 2 ** (bits_per_element - 1)).tolist()

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

    def clean_a_chunk(self, round_idx, chunk_idx):
        self.batch_delete_shared_values(keys=[
            'data', 'agg_res', 'sample_hadamard_seed'
        ],
            postfix=[round_idx, chunk_idx]
        )
        gc.collect()
