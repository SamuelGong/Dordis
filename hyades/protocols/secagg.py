import os
import gc
import logging
import numpy as np
import multiprocessing as mp
from hyades.config import Config
from hyades.protocols import plaintext
from networkx.generators.harary_graph import *
from hyades.protocols.const import SecAggConst
from hyades.utils.misc import plaintext_add, \
    plaintext_aggregate, get_chunks_idx
from hyades.primitives.key_agreement \
    import registry as key_agreement_registry
from hyades.primitives.secret_sharing \
    import registry as secret_sharing_registry
from hyades.primitives.authenticated_encryption \
    import registry as authenticated_encryption_registry
from hyades.primitives.pseudorandom_generator \
    import registry as pseudorandom_generator_registry
from hyades.primitives.utils \
    import rand_bytes, secagg_concatenate, secagg_separate
from hyades.primitives.differential_privacy\
    .utils.misc import modular_clip
from hyades.utils.share_memory_handler \
    import SCHEDULE, NEIGHBORS_DICT, NEIGHBORS

N_CPUS = mp.cpu_count()


# # be careful, this should not be used with SecAgg+
# def get_common_factors(secret_dict_dict, first_client_id):
#     seed_shares_list = []
#     for _, d in secret_dict_dict.items():
#         for idx, seed_share in enumerate(d[first_client_id][1]):  # idx 0 is for b_share
#             if len(seed_shares_list) == idx:
#                 seed_shares_list.append([seed_share])
#             else:
#                 seed_shares_list[idx].append(seed_share)
#
#     first_seed_shares = seed_shares_list[0]
#     secret_sharing_handler = secret_sharing_registry.get()
#     common_factors = secret_sharing_handler.get_factors_for_combine(first_seed_shares)
#     return common_factors
#
#
# # be careful, this should not be used with SecAgg+
# def server_recover_seed_worker(client_id, secret_dict_dict, common_factors):
#     secret_sharing_handler = secret_sharing_registry.get()
#     seed_shares_list = []
#     for _, d in secret_dict_dict.items():
#         for idx, seed_share in enumerate(d[client_id][1]):  # idx 0 is for b_share
#             if len(seed_shares_list) == idx:
#                 seed_shares_list.append([seed_share])
#             else:
#                 seed_shares_list[idx].append(seed_share)
#
#     seeds = []
#     for seed_shares in seed_shares_list:
#         seed = secret_sharing_handler \
#                 .combine_shares(seed_shares, aux=(common_factors,))
#         # big-ended, see around Line 819 in secagg.py
#         seed = int.from_bytes(seed, 'big')
#         seeds.append(seed)
#     return client_id, seeds


def server_generate_output_worker(client_list, recoverable_set,
                                  secret_dict_dict, surviving_clients,
                                  num_range, dim, bit_width,
                                  shared_key_s_dict, public_keys_dict,
                                  neighbors_dict=None):
    result = [0] * dim
    secret_sharing_handler = secret_sharing_registry.get()
    pseudorandom_generator_handler = pseudorandom_generator_registry.get()
    key_agreement_handler = key_agreement_registry.get()
    # debug_prefix = f"[Worker {os.getpid()}]"
    # logging.info(f"{debug_prefix} Processing {len(client_list)} clients.")

    for client_id in client_list:
        surviving_neighbors = surviving_clients
        recoverable_neighbors = recoverable_set
        if neighbors_dict is not None:
            neighbors = neighbors_dict[client_id]
            surviving_neighbors = [e for e in surviving_clients if e in neighbors]
            if client_id in recoverable_set:
                surviving_neighbors.append(client_id)
            else:
                recoverable_neighbors = [e for e in recoverable_set if e in neighbors]

        # logging.info(f"{debug_prefix} [Debug] In server_generate_output_worker:"
        #              f" Start processing client "
        #              f"{client_id} whose neighbors are {neighbors} (surviving: {surviving_neighbors}). "
        #              f"Recoverable_set: {recoverable_set}.")
        if client_id in recoverable_set:
            # logging.info(f"{debug_prefix} [A] Start processing client {client_id}.")
            # if dropout_resilient:
            #     self_mask_share_list = [secret_dict_dict[s][client_id][0]  # idx 1 is for seed_list
            #                             for s in surviving_clients]
            # else:
            self_mask_share_list = [secret_dict_dict[s][client_id]
                                    for s in surviving_neighbors]

            b = secret_sharing_handler \
                .combine_shares(self_mask_share_list)
            pseudorandom_generator_handler.set_seed(b)
            self_mask = pseudorandom_generator_handler \
                .generate_numbers(
                num_range=num_range,
                dim=dim
            )

            self_mask = [-x for x in self_mask]
            result = plaintext_add(
                x=result,
                y=self_mask,
                mod_bit=bit_width
            )
            # logging.info(f"{debug_prefix} [A] End processing client {client_id}.")
        else:
            # logging.info(f"{debug_prefix} [B] Start processing client {client_id}.")
            s_sk_share_list = [secret_dict_dict[s][client_id]
                               for s in surviving_neighbors]
            s_sk_bytes = secret_sharing_handler \
                .combine_shares(s_sk_share_list)
            s_sk = key_agreement_handler\
                .bytes_to_secret_key(s_sk_bytes)
            for c in recoverable_neighbors:
                if c in shared_key_s_dict \
                        and client_id in shared_key_s_dict[c]:
                    # already generate, no need replicate
                    shared_key_s = shared_key_s_dict[c][client_id]
                else:
                    # need to generate
                    s_pk_c_bytes = public_keys_dict[c][1]
                    s_pk_c = key_agreement_handler \
                        .bytes_to_public_key(s_pk_c_bytes)
                    shared_key_s = key_agreement_handler \
                        .generate_shared_key(
                        sk=s_sk,
                        pk=s_pk_c
                    )
                    if c not in shared_key_s_dict:
                        shared_key_s_dict[c] = {}
                    shared_key_s_dict[c][client_id] = shared_key_s

                    if client_id not in shared_key_s_dict:
                        shared_key_s_dict[client_id] = {}
                    shared_key_s_dict[client_id][c] = shared_key_s

                pseudorandom_generator_handler \
                    .set_seed(seed=shared_key_s)
                pairwise_mask = pseudorandom_generator_handler \
                    .generate_numbers(
                    num_range=num_range,
                    dim=dim
                )
                if client_id > c:
                    pairwise_mask = [-x for x in pairwise_mask]
                result = plaintext_add(
                    x=result,
                    y=pairwise_mask,
                    mod_bit=bit_width
                )
            # logging.info(f"{debug_prefix} [B] End processing client {client_id}.")

    return result


class ProtocolServer(plaintext.ProtocolServer, SecAggConst):
    def __init__(self, client_id=0):
        SecAggConst.__init__(self)
        plaintext.ProtocolServer.__init__(self, client_id=client_id)
        self.bit_width = Config().agg.security.bit_width

        self.secret_sharing_handler = secret_sharing_registry.get()
        self.pseudorandom_generator_handler = pseudorandom_generator_registry.get()
        self.key_agreement_handler = key_agreement_registry.get()

        if hasattr(Config().clients, "dropout"):
            if hasattr(Config().clients.dropout, "seed"):
                seed = Config().clients.dropout.seed
                np.random.seed(seed)
            self.dropout_mocking_phase = self.UPLOAD_DATA

        # added for SecAgg+
        self.advanced = False
        if hasattr(Config().agg.security, "advanced") \
                and Config().agg.security.advanced:
            self.advanced = True
            logging.info(f"SecAgg+ enabled.")

    def set_graph_dict(self):
        self.graph_dict = {
            self.PREPARE_DATA: {
                "worker": "prepare_data",
                "stage": self._no_plot_phase_stage_mapping[
                    self.PREPARE_DATA],
                "predecessor_ids": []
            },
            self.ENCODE_DATA: {
                "worker": "encode_data",
                "stage": self._no_plot_phase_stage_mapping[
                    self.ENCODE_DATA],
                "predecessor_ids": [self.PREPARE_DATA]
            },
            self.SHARE_KEYS: {
                "worker": "share_keys",
                "stage": self._no_plot_phase_stage_mapping[
                    self.SHARE_KEYS],
                "predecessor_ids": [self.ADVERTISE_KEYS]
            },
            self.MASKING: {
                "worker": "masking",
                "stage": self._no_plot_phase_stage_mapping[
                    self.MASKING],
                "predecessor_ids": [self.ENCODE_DATA, self.SHARE_KEYS]
            },
            self.UPLOAD_DATA: {
                "worker": "upload_data",
                "stage": self._no_plot_phase_stage_mapping[
                    self.UPLOAD_DATA],
                "predecessor_ids": [self.MASKING]
            },
            self.UNMASKING: {
                "worker": "unmasking",
                "stage": self._no_plot_phase_stage_mapping[
                    self.UNMASKING],
                "predecessor_ids": [self.UPLOAD_DATA]
            },
            self.GENERATE_OUTPUT: {
                "worker": "generate_output",
                "stage": self._no_plot_phase_stage_mapping[
                    self.GENERATE_OUTPUT],
                "predecessor_ids": [self.UNMASKING]
            },
            self.SERVER_USE_OUTPUT: {
                "worker": "server_use_output",
                "stage": self._no_plot_phase_stage_mapping[
                    self.SERVER_USE_OUTPUT],
                "predecessor_ids": [self.GENERATE_OUTPUT],
            },
            self.DOWNLOAD_DATA: {
                "worker": "download_data",
                "stage": self._no_plot_phase_stage_mapping[
                    self.DOWNLOAD_DATA],
                "predecessor_ids": [self.SERVER_USE_OUTPUT],
            },
            self.DECODE_DATA: {
                "worker": "decode_data",
                "stage": self._no_plot_phase_stage_mapping[
                    self.DECODE_DATA],
                "predecessor_ids": [self.DOWNLOAD_DATA],
            },
            self.CLIENT_USE_OUTPUT: {
                "worker": "client_use_output",
                "stage": self._no_plot_phase_stage_mapping[
                    self.CLIENT_USE_OUTPUT],
                "predecessor_ids": [self.DECODE_DATA],
                "is_final": True
            },
        }

        node_1 = {
            "worker": "advertise_keys",
            "stage": self._no_plot_phase_stage_mapping[
                self.ADVERTISE_KEYS],
        }

        if hasattr(Config().agg, "parallel") and Config().agg.parallel:
            node_1.update({
                "predecessor_ids": []
            })
        else:
            node_1.update({
                "predecessor_ids": [self.ENCODE_DATA]
            })

        self.graph_dict.update({
            self.ADVERTISE_KEYS: node_1
        })

    def get_threshold(self, round_idx):
        if round_idx not in self.threshold_dict:
            num_sampled_clients \
                = self.get_num_sampled_clients(round_idx=round_idx)
            threshold_frac = Config().agg.security.secret_sharing.threshold \
                if hasattr(Config().agg.security.secret_sharing, "threshold") else 1.0
            threshold = int(np.ceil(threshold_frac * num_sampled_clients))
            self.threshold_dict[round_idx] = threshold
        else:
            threshold = self.threshold_dict[round_idx]

        return threshold

    def store_client_payload(self, args):
        client_id, payload, round_idx, chunk_idx, phase_idx = args

        if phase_idx == self.ENCODE_DATA \
                or phase_idx == self.MASKING \
                or phase_idx == self.DOWNLOAD_DATA \
                or phase_idx == self.DECODE_DATA \
                or phase_idx == self.CLIENT_USE_OUTPUT:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=1
            )
        elif phase_idx == self.PREPARE_DATA:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['meta']
            )
        elif phase_idx == self.ADVERTISE_KEYS:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['public_keys']
            )
        elif phase_idx == self.SHARE_KEYS:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['shared_secret']
            )
        elif phase_idx == self.UPLOAD_DATA:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['data']
            )
        elif phase_idx == self.UNMASKING:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['info']
            )
        else:
            raise ValueError(f"Unknown phase: {phase_idx}.")

        self.threshold_test(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )

    def advertise_keys(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.ADVERTISE_KEYS
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        logging.info("%s Phase started. Instructing all "
                     "clients to advertise keys.", log_prefix_str)

        sampled_clients = self.fast_get_sampled_clients(round_idx)
        result = {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
            },
            "send_list": sampled_clients,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }

        if self.advanced:
            # TODO: to determined k automatically
            k = Config().agg.security.advanced.k
            net = hkn_harary_graph(
                k, len(sampled_clients)
            )

            logical_neighbors_dict = {}
            for idx in range(len(sampled_clients)):
                client_id = sampled_clients[idx]
                n_list = [sampled_clients[e[1]] for e in net.edges(idx)]
                logical_neighbors_dict[client_id] = n_list

            self.set_a_shared_value(
                key=[NEIGHBORS_DICT, round_idx],
                value=logical_neighbors_dict
            )
            logging.info(f"{log_prefix_str} "
                         f"Neighbors dict for SecAgg+: "
                         f"{logical_neighbors_dict}.")
            result["payload"]["neighbors_dict"] = logical_neighbors_dict

        return result

    def share_keys(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.SHARE_KEYS
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        data = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.ADVERTISE_KEYS
        )
        surviving_clients = sorted(list(data.keys()))
        logging.info("%s Phase started. Instructing %d clients to "
                     "share keys: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'data': data,
                'phase': phase_idx,
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "send_list": surviving_clients,
            "log_prefix_str": log_prefix_str
        }

    def masking(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.MASKING
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        shared_keys, prepared_data \
            = self.get_records_for_phases(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phases=[self.SHARE_KEYS, self.ENCODE_DATA]
        )
        # should be the intersection!
        surviving_clients = sorted(list(set(list(shared_keys.keys()))\
            .intersection(
            set(list(prepared_data.keys()))
        )))
        logging.info("%s Phase started. Instructing %d clients to "
                     "mask data: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)
        self.set_a_shared_value(
            key=['u_2', round_idx, chunk_idx],
            value=surviving_clients
        )

        exchange_dict = {}
        for src_client_id, shared_secret in shared_keys.items():
            for dst_client_id, share_dict in shared_secret.items():
                if dst_client_id not in surviving_clients:
                    continue
                if dst_client_id not in exchange_dict:
                    exchange_dict[dst_client_id] = {
                        'round': round_idx,
                        'chunk': chunk_idx,
                        'phase': phase_idx,
                        'data': {}
                    }
                exchange_dict[dst_client_id]['data'].update({
                    src_client_id: share_dict
                })

        return {
            "payload": exchange_dict,
            "round_idx": round_idx,
            "send_type": "exchange",
            "log_prefix_str": log_prefix_str,
            "key_postfix": [round_idx, chunk_idx, phase_idx]
        }

    def unmasking(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.UNMASKING
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        data = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.UPLOAD_DATA
        )
        surviving_clients = sorted(list(data.keys()))
        self.set_a_shared_value(
            key=["u_3", round_idx, chunk_idx],
            value=surviving_clients
        )
        logging.info("%s Phase started. Instructing %d clients to "
                     "provide unmasking help: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'surviving_clients': surviving_clients,
                'phase': phase_idx,
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "send_list": surviving_clients,
            "log_prefix_str": log_prefix_str
        }

    def generate_output(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.GENERATE_OUTPUT
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        shared_key_s_dict = {}  # internal use

        u_2, u_3 = self.batch_get_shared_values(
            keys=['u_2', 'u_3'],
            postfix=[round_idx, chunk_idx]
        )
        u_2_minus_u_3 = list(set(u_2) - set(u_3))
        self.update_stat(
            clients=u_3,  # TODO: maybe u_2?
            log_prefix_str=log_prefix_str,
            round_idx=round_idx,
            chunk_idx=chunk_idx
        )
        logging.info("%s Phase started. Generating output with U_2: %s, "
                     "U_3: %s, U2/U3: %s.", log_prefix_str,
                     u_2, u_3, u_2_minus_u_3)

        public_keys_dict, masked_data_dict, info_dict_dict \
            = self.get_records_for_phases(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phases=[
                self.ADVERTISE_KEYS,
                self.UPLOAD_DATA,
                self.UNMASKING
            ]
        )
        secret_dict_dict = {k: v["secret_dict"]
                            for k, v in info_dict_dict.items()}

        if hasattr(Config().agg, "differential_privacy") \
                and hasattr(Config().agg.differential_privacy,
                   "dropout_resilience"):
            u_4 = list(info_dict_dict.keys())
            recoverable_set = u_4
            logging.info(f"Recoverable list of clients is U4: "
                         f"{recoverable_set}.")
        else:
            recoverable_set = u_3
            logging.info(f"Recoverable list of clients is U3: "
                         f"{recoverable_set}.")

        # unbatch first, if necessary
        masked_data_list = [masked_data_dict[client_id]
                            for client_id in recoverable_set]

        if Config().agg.type in ["secagg", "dp_plus_secagg"] \
                and hasattr(Config().agg, "quantize") \
                and hasattr(Config().agg.quantize, "batch"):
            if Config().agg.type == "secagg":
                bits_per_element = Config().agg.security.bit_width
            else:  # type == "dp_plus_secagg"
                bits_per_element = self.dp_handler.get_bits()
            batching_params = Config().agg.quantize.batch

            unbatched_masked_data_list = []
            for masked_data in masked_data_list:
                original_length = self.chunk_size[chunk_idx]
                if hasattr(Config().agg, "differential_privacy"):
                    original_length = self.dp_handler.get_padded_dim(
                        dim=original_length
                    )

                unbatched_masked_data = self.unbatch_data(
                    data=masked_data,
                    batching_params=batching_params,
                    bits_per_element=bits_per_element,
                    original_length=original_length,
                    log_prefix_str=log_prefix_str
                )
                unbatched_masked_data_list.append(unbatched_masked_data)
            masked_data_list = unbatched_masked_data_list

        # do it early to save space
        self.delete_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.UPLOAD_DATA
        )

        # aggregate with masks (and excessive noise)

        # for masked_data in masked_data_list:
        #     logging.info(f"[Debug] {masked_data[:6]} {masked_data[-6:]}.")
        masked_agg_res = plaintext_aggregate(
            data_list=masked_data_list,
            mod_bit=self.bit_width
        )
        logging.info(f"[Debug] Aggregation result:"
                     f"first 6: {masked_agg_res[:6]},"
                     f"last 6: {masked_agg_res[-6:]}.")

        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == 'simple'):
            # reconstruct masks and unmask the aggregate
            logging.info(f"{log_prefix_str} Start to unmask the aggregate.")

            num_range = [0, 1 << self.bit_width]
            dim = len(masked_agg_res)
            surviving_clients = list(secret_dict_dict.keys())
            processes = N_CPUS

            # balancing the workload on a finer granularity
            # otherwise the performance can have great standard deviation

            # for scalability. TODO: avoid hard-coding
            # pool_outputs = None
            # for b in range(0, len(recoverable_set), 16):
            #     e = b + 16
            #     if e > len(recoverable_set):
            #         e = len(recoverable_set)
            #     recoverable_subset = recoverable_set[b:e]

            u_2_in = [e for e in u_2 if e in recoverable_set]
            u_2_out = [e for e in u_2 if e not in recoverable_set]
            u_2_in_list = [u_2_in[begin:end]
                           for begin, end in get_chunks_idx(len(u_2_in), processes)]
            u_2_out_list = [u_2_out[begin:end]
                            for begin, end in get_chunks_idx(len(u_2_out), processes)]
            u_2_list = [a + b for a, b in zip(u_2_in_list, u_2_out_list)]

            # if hasattr(Config().agg, "differential_privacy") \
            #         and hasattr(Config().agg.differential_privacy,
            #                     "dropout_resilience"):  # new
            #     pool_inputs = [(u_2_part, recoverable_set,
            #                     secret_dict_dict, surviving_clients,
            #                     num_range, dim, self.bit_width,
            #                     shared_key_s_dict, public_keys_dict, True)
            #                    for u_2_part in u_2_list]
            # else:
            if not self.advanced:
                pool_inputs = [(u_2_part, recoverable_set,
                                secret_dict_dict, surviving_clients,
                                num_range, dim, self.bit_width,
                                shared_key_s_dict, public_keys_dict, None)
                               for u_2_part in u_2_list]
            else:
                neighbors_dict = self.fast_get_neighbors_dict(round_idx)
                pool_inputs = [(u_2_part, recoverable_set,
                                secret_dict_dict, surviving_clients,
                                num_range, dim, self.bit_width,
                                shared_key_s_dict, public_keys_dict, neighbors_dict)
                               for u_2_part in u_2_list]

            with mp.Pool(processes=processes) as pool:
                pool_outputs = pool.starmap(
                    server_generate_output_worker, pool_inputs)

            for negative_mask in pool_outputs:
                masked_agg_res = plaintext_add(
                    x=masked_agg_res,
                    y=negative_mask,
                    mod_bit=self.bit_width
                )

            logging.info(f"{log_prefix_str} The aggregate has been unmasked.")

        # eliminate excessive noise
        if hasattr(Config().agg, "differential_privacy"):
            # though batching is not supported in dp_plus_secagg
            # we still have made it positive
            # so we need to undo this
            bits_per_element = self.dp_handler.get_bits()
            masked_agg_res = np.array(masked_agg_res) \
                             - len(recoverable_set) * 2 ** (bits_per_element - 1)
            dp_params_dict = self.get_a_shared_value(
                key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
            )

            if hasattr(Config().agg.differential_privacy, "dropout_resilience") \
                and (Config().agg.differential_privacy
                             .dropout_resilience.type == "simple"
                     or Config().agg.differential_privacy
                             .dropout_resilience.type == "log2"):

                # if False:  # Only for testing
                # if (hasattr(Config(), "simulation")
                #         and Config().simulation.type == 'simple'):
                excessive_noise_seeds_dict = {k: v["excessive_noise_seeds"]
                                              for k, v in info_dict_dict.items()}
                # else:
                #     # for acceleration
                #     # logging.info(f"[Debug] here")
                #     first_client_id = recoverable_set[0]
                #     common_factors = get_common_factors(
                #         secret_dict_dict=secret_dict_dict,
                #         first_client_id=first_client_id
                #     )
                #     # logging.info(f"[Debug] {common_factors is not None}")
                #
                #     pool_inputs = [
                #         (client_id, secret_dict_dict, common_factors)
                #         for client_id in recoverable_set
                #     ]
                #
                #     processes = N_CPUS
                #     with mp.Pool(processes=processes) as pool:
                #         pool_outputs = pool.starmap(
                #             server_recover_seed_worker, pool_inputs)
                #
                #     excessive_noise_seeds_dict = {}
                #     # unlike what in simulation mode, the resulting excessive_noise_seeds_dict
                #     # does not contain the noise level of the components
                #     # hence we need to combine them
                #     for client_id, excessive_noise_seeds in pool_outputs:
                #         excessive_noise_seeds_dict[client_id] = list(zip(
                #             excessive_noise_seeds,
                #             info_dict_dict[client_id]["excessive_noise_seeds"],
                #         )) # seed isqat idx 0 and stddev is at idx 1

                # logging.info(f"{log_prefix_str} seeds recovered.")
                # if need to drop excessive noise
                if excessive_noise_seeds_dict[
                    list(excessive_noise_seeds_dict.keys())[0]]:

                    dim = len(masked_agg_res)
                    xnoise_params = {}  # server does not need to specify this
                    pool_inputs = [([0] * dim, xnoise_params, log_prefix_str,
                                    execessive_noise_seeds, True)
                                   for client_id, execessive_noise_seeds
                                   in excessive_noise_seeds_dict.items()]

                    # pool_outputs = None
                    # for b in range(0, len(pool_inputs), 16):
                    #     e = b + 16
                    #     if e > len(pool_inputs):
                    #         e = len(pool_inputs)
                    #     pool_inputs_subset = pool_inputs[b:e]

                    processes = N_CPUS // 2  # based on practical observations
                    with mp.Pool(processes=processes) as pool:
                        pool_outputs = pool.starmap(
                            # self.dp_handler.add_excessive_noise, pool_inputs_subset)
                            self.dp_handler.add_excessive_noise, pool_inputs)
                    for negative_noise, _ in pool_outputs:
                        masked_agg_res = plaintext_add(
                            x=masked_agg_res,
                            y=negative_noise,
                        )

            masked_agg_res = modular_clip(
                data=masked_agg_res,
                log_prefix_str=log_prefix_str,
                other_args=(dp_params_dict,)
            )

        self.batch_set_shared_values(d={
            'agg_res': masked_agg_res,
            'involved_clients': recoverable_set
        },
            postfix=[round_idx, chunk_idx])
        self._publish_a_value(
            channel=SCHEDULE,
            message=[round_idx, chunk_idx, phase_idx]
        )
        logging.info("%s Phase done.", log_prefix_str)
        return {}

    def clean_a_chunk(self, round_idx, chunk_idx):
        """ Called at plaintext.use_output. """
        self.batch_delete_shared_values(
            keys=['agg_res', 'involved_clients'],
            postfix=[round_idx, chunk_idx]
        )
        if hasattr(Config().agg, "differential_privacy"):
            self.delete_a_shared_value(
                key=["sample_hadamard_seed", round_idx, chunk_idx]
            )

        self.delete_records_for_phases(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phases=[self.PREPARE_DATA, self.ENCODE_DATA,
                    self.ADVERTISE_KEYS, self.SHARE_KEYS,
                    # self.UPLOAD_DATA, self.MASKING,
                    self.MASKING,
                    self.UNMASKING, self.DOWNLOAD_DATA,
                    self.DECODE_DATA, self.CLIENT_USE_OUTPUT]
        )
        self.clients_dropped_out = []
        gc.collect()


class ProtocolClient(plaintext.ProtocolClient, SecAggConst):
    def __init__(self, client_id):
        SecAggConst.__init__(self)
        plaintext.ProtocolClient.__init__(self, client_id)
        self.ss_threshold_frac \
            = Config().agg.security.secret_sharing.threshold
        self.bit_width = Config().agg.security.bit_width

        self.key_agreement_handler = key_agreement_registry.get()
        self.secret_sharing_handler = secret_sharing_registry.get()
        self.authenticated_encryption_handler \
            = authenticated_encryption_registry.get()
        self.pseudorandom_generator_handler \
            = pseudorandom_generator_registry.get()

        # added for SecAgg+
        self.advanced = False
        if hasattr(Config().agg.security, "advanced") \
                and Config().agg.security.advanced:
            self.advanced = True
            logging.info(f"SecAgg+ enabled.")

    def get_threshold(self, round_idx):
        num_sampled_clients = self.get_num_sampled_clients(round_idx=round_idx)
        return int(np.ceil(self.ss_threshold_frac * num_sampled_clients))

    def set_routine(self):
        self.routine = {
            self.PREPARE_DATA: "prepare_data",
            self.ENCODE_DATA: "encode_data",
            self.ADVERTISE_KEYS: "advertise_keys",
            self.SHARE_KEYS: "share_keys",
            self.MASKING: "masking",
            self.UPLOAD_DATA: "upload_data",
            self.UNMASKING: "unmasking",
            self.DOWNLOAD_DATA: "download_data",
            self.DECODE_DATA: "decode_data",
            self.CLIENT_USE_OUTPUT: "client_use_output"
        }

    def advertise_keys(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        if self.advanced:
            neighbors_dict = payload["neighbors_dict"]
            neighbors = neighbors_dict[logical_client_id]
            self.set_a_shared_value(
                key=[NEIGHBORS, round_idx],
                value=neighbors
            )
            logging.info(f"Received my neighbors for SecAgg+:"
                         f" {sorted(neighbors)}.")

        response = {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                "public_keys": (1, 1),  # placeholder
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Key uploaded."
        }

        # if True:  # Only for testing
        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == 'simple'):
            c_keypair = self.key_agreement_handler.generate_key_pairs()
            s_keypair = self.key_agreement_handler.generate_key_pairs()
            c_private_key_bytes = self.key_agreement_handler \
                .secret_key_to_bytes(sk=c_keypair[0])
            c_public_key_bytes = self.key_agreement_handler \
                .public_key_to_bytes(pk=c_keypair[1])
            s_private_key_bytes = self.key_agreement_handler \
                .secret_key_to_bytes(sk=s_keypair[0])
            s_public_key_bytes = self.key_agreement_handler \
                .public_key_to_bytes(pk=s_keypair[1])
            self.batch_set_shared_values(
                d={
                    'c_keypair': (c_private_key_bytes, c_public_key_bytes),
                    's_keypair': (s_private_key_bytes, s_public_key_bytes)
                },
                postfix=[round_idx, chunk_idx]
            )
            response["payload"]["public_keys"] \
                = (c_public_key_bytes, s_public_key_bytes)

        return response

    def share_keys(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        surviving_client_dict = payload['data']
        share_dict = {}
        response = {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                "shared_secret": {},
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Secret shared."
        }

        if self.advanced:
            neighbors = self.get_a_shared_value(key=[NEIGHBORS, round_idx])

            # only keep me and neighbors
            keys_to_delete = []
            for client_id in surviving_client_dict.keys():
                if client_id == logical_client_id:
                    continue
                if client_id not in neighbors:
                    keys_to_delete.append(client_id)
            for key in keys_to_delete:
                del surviving_client_dict[key]
            # logging.info(f"{log_prefix_str} [Debug] N {neighbors}, surviving in share_keys "
            #              f"(except self): {surviving_client_dict.keys()}.")

        # if False:  # Only for testing
        if (hasattr(Config(), "simulation")
                and Config().simulation.type == 'simple'):
            for dst_client_id, _ in surviving_client_dict.items():
                if dst_client_id == logical_client_id:
                     continue
                share_dict[dst_client_id] = 1  # placeholder
        else:
            # sample a random element b
            b = rand_bytes(num=32)
            if not self.advanced:
                t = self.get_threshold(round_idx=round_idx)
                n = len(list(surviving_client_dict.keys()))  # i.e., U_1
            else:
                n = len(list(surviving_client_dict.keys()))
                # TODO: currently hard-coding according to Olympia
                # resulting in a beta = t/k = 1/2
                t = n // 2
            logging.info(f"{log_prefix_str}: t={t} "
                         f"and n={n} for secret sharing.")

            # generate t-out-of-U_1 shares for s_sk and b
            c_keypair, s_keypair = self.batch_get_shared_values(
                keys=['c_keypair', 's_keypair'],
                postfix=[round_idx, chunk_idx]
            )
            s_secret_key_bytes = s_keypair[0]
            s_sk_share_list = self.secret_sharing_handler.create_shares(
                secret=s_secret_key_bytes, t=t, n=n
            )
            b_share_list = self.secret_sharing_handler.create_shares(
                secret=b, t=t, n=n
            )

            if hasattr(Config().agg, "differential_privacy") \
                    and hasattr(Config().agg.differential_privacy,
                                "dropout_resilience"):  # new
                excessive_noise_seeds = self.get_a_shared_value(
                    key=["excessive_noise_seeds", round_idx, chunk_idx]
                )
                seed_list_share_dict = {}
                for seed in excessive_noise_seeds:
                    # idx 0 is seed and 1 is stddev
                    # each seed should be 32 bits (4 bytes), see dskellam.py
                    seed_shares = self.secret_sharing_handler.create_shares(
                        secret=seed[0].to_bytes(32, 'big'), t=t, n=n
                    )
                    for client_idx in range(n):
                        if client_idx not in seed_list_share_dict:
                            seed_list_share_dict[client_idx] = []
                        seed_list_share_dict[client_idx].append(
                            seed_shares[client_idx]
                        )

            client_cnt = -1
            client_data_dict = {}
            self_b_share = None
            self_s_sk_share = None
            if hasattr(Config().agg, "differential_privacy") \
                    and hasattr(Config().agg.differential_privacy,
                                "dropout_resilience"):  # new
                self_seed_list_share = None

            for dst_client_id, data in surviving_client_dict.items():
                client_cnt += 1
                if dst_client_id == logical_client_id:
                    self_b_share \
                        = b_share_list[client_cnt]
                    self_s_sk_share \
                        = s_sk_share_list[client_cnt]
                    if hasattr(Config().agg, "differential_privacy") \
                            and hasattr(Config().agg.differential_privacy,
                                        "dropout_resilience"):  # new
                        self_seed_list_share \
                            = seed_list_share_dict[client_cnt]
                    continue

                c_pk_bytes, s_pk_bytes = data
                c_pk = self.key_agreement_handler.bytes_to_public_key(c_pk_bytes)
                self_c_sk_bytes = c_keypair[0]
                self_c_sk = self.key_agreement_handler.bytes_to_secret_key(
                    bytes=self_c_sk_bytes
                )
                shared_key_c = self.key_agreement_handler.generate_shared_key(
                    sk=self_c_sk,
                    pk=c_pk
                )
                client_data_dict[dst_client_id] = {
                    'c_pk': c_pk_bytes,
                    's_pk': s_pk_bytes,
                    'shared_key_c': shared_key_c
                }

                self.authenticated_encryption_handler.set_key(shared_key_c)
                b_share = b_share_list[client_cnt]
                s_sk_share = s_sk_share_list[client_cnt]
                if hasattr(Config().agg, "differential_privacy") \
                        and hasattr(Config().agg.differential_privacy,
                                    "dropout_resilience"):  # new
                    seed_list_share = seed_list_share_dict[client_cnt]
                    message = secagg_concatenate(
                        logical_client_id, dst_client_id, b_share,
                        s_sk_share, seed_list_share
                    )
                else:
                    message = secagg_concatenate(
                        logical_client_id, dst_client_id,
                        b_share, s_sk_share
                    )
                encrypted_message = \
                    self.authenticated_encryption_handler.encrypt(message)
                share_dict[dst_client_id] = encrypted_message

            d = {
                'b': b,
                'b_share': self_b_share,
                's_sk_share': self_s_sk_share,
                'client_data': client_data_dict
            }
            if hasattr(Config().agg, "differential_privacy") \
                    and hasattr(Config().agg.differential_privacy,
                                "dropout_resilience"):  # new
                d['seed_list_share'] = self_seed_list_share

            self.batch_set_shared_values(
                d=d,
                postfix=[round_idx, chunk_idx]
            )

        response["payload"]["shared_secret"] = share_dict
        return response

    def masking(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        response = {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Data masked."
        }

        surviving_client_dict = payload['data']
        u_2 = []
        data = self.get_a_shared_value(
            key=['data', round_idx, chunk_idx]
        )

        # if False:  # only for testing
        if (hasattr(Config(), "simulation")
                and Config().simulation.type == 'simple'):
            for src_client_id, _ \
                    in surviving_client_dict.items():
                u_2.append(src_client_id)
            u_2.append(logical_client_id)
        else:
            b, s_keypair, client_data_dict \
                = self.batch_get_shared_values(
                keys=['b', 's_keypair', 'client_data'],
                postfix=[round_idx, chunk_idx]
            )

            for src_client_id, encrypted_message \
                    in surviving_client_dict.items():
                # there must be that src_client_id != logical_client_id
                client_data_dict[src_client_id]['encrypted_message'] \
                    = encrypted_message

                s_pk_bytes = client_data_dict[src_client_id]['s_pk']
                s_pk = self.key_agreement_handler.bytes_to_public_key(
                    bytes=s_pk_bytes
                )
                self_s_sk_bytes = s_keypair[0]
                self_s_sk = self.key_agreement_handler.bytes_to_secret_key(
                    bytes=self_s_sk_bytes
                )
                shared_key_s = self.key_agreement_handler.generate_shared_key(
                    sk=self_s_sk,
                    pk=s_pk
                )
                client_data_dict[src_client_id]['shared_key_s'] \
                    = shared_key_s
                u_2.append(src_client_id)
            u_2.append(logical_client_id)

            # logging.info(f"{log_prefix_str} [Debug] "
            #              f"U_2 in masking (excluding self): {surviving_client_dict.keys()}.")

            # mask data
            num_range = [0, 1 << self.bit_width]
            dim = len(data)

            surviving_client_list = list(surviving_client_dict.keys())
            # i.e., U_2 except for the client itself

            for src_client_id in surviving_client_list:
                shared_key_s = client_data_dict[src_client_id]['shared_key_s']
                self.pseudorandom_generator_handler.set_seed(shared_key_s)
                # if False:  # debug only
                pairwise_mask = self.pseudorandom_generator_handler\
                    .generate_numbers(
                    num_range=num_range,
                    dim=dim
                )
                if src_client_id < logical_client_id:
                    pairwise_mask = [-x for x in pairwise_mask]

                data = plaintext_add(
                    x=pairwise_mask,
                    y=data,
                    mod_bit=self.bit_width
                )

            self.pseudorandom_generator_handler.set_seed(seed=b)
            # if False:  # debug only
            self_mask = self.pseudorandom_generator_handler\
                .generate_numbers(
                num_range=num_range,
                dim=dim
            )
            data = plaintext_add(
                x=self_mask,
                y=data,
                mod_bit=self.bit_width
            )

            self.set_a_shared_value(
                key=['client_data', round_idx, chunk_idx],
                value=client_data_dict
            )

        if Config().agg.type in ["secagg", "dp_plus_secagg"] \
                and hasattr(Config().agg, "quantize") \
                and hasattr(Config().agg.quantize, "batch"):
            batching_params = Config().agg.quantize.batch
            if Config().agg.type == "secagg":
                bits_per_element = Config().agg.security.bit_width
            else:  # type == "dp_plus_secagg"
                bits_per_element = self.dp_handler.get_bits()
            data = self.batch_data(
                data=data,
                batching_params=batching_params,
                bits_per_element=bits_per_element,
                log_prefix_str=log_prefix_str
            )

        self.batch_set_shared_values(d={
            'u_2': u_2,
            'data': data
        }, postfix=[round_idx, chunk_idx])
        return response

    def unmasking(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        u_3 = payload['surviving_clients']
        if self.advanced:
            neighbors = self.get_a_shared_value(key=[NEIGHBORS, round_idx])

            # only keep me and neighbors
            keys_to_reserve = []
            for client_id in u_3:
                if client_id == logical_client_id \
                        or client_id in neighbors:
                    keys_to_reserve.append(client_id)
            u_3 = keys_to_reserve

        u_2 = self.get_a_shared_value(key=['u_2', round_idx, chunk_idx])

        u_2_minus_u_3 = list(set(u_2) - set(u_3))
        logging.info("%s U_2: {%s}, U_3: {%s}, U2/U3: {%s}.",
                     log_prefix_str, u_2, u_3, u_2_minus_u_3)

        excessive_noise_seeds = None
        component_idx_to_deduct = None
        if hasattr(Config().agg, "differential_privacy") \
                and hasattr(Config().agg.differential_privacy,
                            "dropout_resilience"):
            excessive_noise_seeds = self.get_a_shared_value(
                key=["excessive_noise_seeds", round_idx, chunk_idx]
            )
            dp_params_dict = self.get_a_shared_value(
                key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx],
            )
            local_stddev = dp_params_dict["local_stddev"]
            num_sampled_clients = self.get_num_sampled_clients(round_idx=round_idx)

            # this is needed for compatibility with trace-driven
            target_num_clients = dp_params_dict["target_num_clients"]
            requried_central_stddev = local_stddev * np.sqrt(target_num_clients)
            actual_baseline_local_stddev = requried_central_stddev / np.sqrt(num_sampled_clients)

            _type = Config().agg.differential_privacy\
                .dropout_resilience.type
            num_dropped_clients = num_sampled_clients - len(u_3)

            if _type == "simple":
                component_idx_to_deduct = range(
                    0, len(excessive_noise_seeds) - num_dropped_clients
                )
            elif _type == "log2":
                xnoise_params = self.dp_handler.get_xnoise_params(
                    round_idx=round_idx,
                    num_sampled_clients=num_sampled_clients,
                    dp_params_dict=dp_params_dict
                )
                dropout_tolerated = xnoise_params["dropout_tolerated"]
                noise_min_var = xnoise_params["noise_min_var"]
                num_noise_levels = xnoise_params["num_noise_levels"]

                local_var_to_deduct = num_sampled_clients * actual_baseline_local_stddev ** 2 * (
                        1. / (num_sampled_clients - dropout_tolerated) - 1.
                        / (num_sampled_clients - num_dropped_clients)
                )
                units_to_deduct = int(np.floor(local_var_to_deduct / noise_min_var))

                if units_to_deduct > 0:
                    component_idx_to_deduct = [0]  # at least to deduct this
                    tmp = units_to_deduct - 1
                    for _idx in range(num_noise_levels):
                        if tmp & 1 == 1:
                            component_idx_to_deduct.append(_idx + 1)
                        tmp >>= 1
                else:  # can be nothing to deduct, e.g., exactly the tolerated amount of dropout
                    component_idx_to_deduct = []

                logging.info(f"[Debug] required_central_stddev: {requried_central_stddev}, "
                             f"actual_baseline_local_stddev: {actual_baseline_local_stddev}, "
                             f"num_sampled_clients: {num_sampled_clients}, "
                             f"num_dropped_clients: {num_dropped_clients}, "
                             f"dropout_tolerated: {dropout_tolerated}, "
                             f"noise_min_var: {noise_min_var}, "
                             f"num_noise_levels: {num_noise_levels}, "
                             f"local_var_to_deduct: {local_var_to_deduct}, "
                             f"units_to_deduct: {units_to_deduct}, "
                             f"component_idx_to_deduct: {component_idx_to_deduct}.")
            else:
                raise NotImplementedError

            # only in simulation mode shall we ignore the security
            # and directly send the seeds to the server
            # if not in simulation mode, the seeds is send in a secret-share manner
            # if False: # only for testing
            # if (hasattr(Config(), "simulation")
            #         and Config().simulation.type == 'simple'):
            # whether simulation or not, first directly send
            excessive_noise_seeds = [excessive_noise_seeds[_idx]
                                     for _idx in component_idx_to_deduct]
            # else:  # conceal the seed otherwise (but reveal the stddev)
            #     excessive_noise_seeds = [excessive_noise_seeds[_idx][1]
            #                              for _idx in component_idx_to_deduct]

        response = {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'info': {
                    'secret_dict': {},  # placeholder
                    'excessive_noise_seeds': excessive_noise_seeds
                },
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Help provided."
        }

        # if True:  # only for testing
        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == 'simple'):
            self_b_share, client_data \
                = self.batch_get_shared_values(
                keys=['b_share', 'client_data'],
                postfix=[round_idx, chunk_idx]
            )
            # if hasattr(Config().agg, "differential_privacy") \
            #         and hasattr(Config().agg.differential_privacy,
            #                     "dropout_resilience"):  # new
            #     # TODO: send the correct number
            #     self_seed_list_share = self.get_a_shared_value(
            #         key=["seed_list_share", round_idx, chunk_idx]
            #     )

            # decrypt secret shares in U_2
            for client_id in u_2:
                if client_id == logical_client_id:
                    continue

                encrypted_message = client_data[
                    client_id]['encrypted_message']
                shared_key_c = client_data[client_id]['shared_key_c']
                self.authenticated_encryption_handler.set_key(shared_key_c)
                concatenated_message \
                    = self.authenticated_encryption_handler.decrypt(encrypted_message)

                if hasattr(Config().agg, "differential_privacy") \
                        and hasattr(Config().agg.differential_privacy,
                                    "dropout_resilience"):  # new
                    src_client_id, dst_client_id, others_b_share, \
                        others_s_sk_share, others_seed_list_share \
                        = secagg_separate(concatenated_message)
                    # though we do not use others_seed_list_share currently
                else:
                    src_client_id, dst_client_id, others_b_share, others_s_sk_share \
                        = secagg_separate(concatenated_message)
                assert src_client_id == client_id
                assert dst_client_id == logical_client_id

                client_data[client_id]['b_share'] = others_b_share
                client_data[client_id]['s_sk_share'] = others_s_sk_share
                # if hasattr(Config().agg, "differential_privacy") \
                #         and hasattr(Config().agg.differential_privacy,
                #                     "dropout_resilience"):
                #     # should send the correct number of seeds
                #     client_data[client_id]['seed_list_share'] = [
                #         others_seed_list_share[_idx] for _idx in component_idx_to_deduct
                #     ]

            # send to server the necessary secrets
            secret_dict = {}
            for client_id in u_2:
                if client_id in u_3:
                    # if hasattr(Config().agg, "differential_privacy") \
                    #         and hasattr(Config().agg.differential_privacy,
                    #                     "dropout_resilience"):  # new
                    #     if client_id == logical_client_id:  # logical_client_id can only go here
                    #         self_seed_list_share = [
                    #             self_seed_list_share[_idx] for _idx in component_idx_to_deduct
                    #         ]
                    #         secret_dict[client_id] = (self_b_share, self_seed_list_share)
                    #     else:
                    #         secret_dict[client_id] = (
                    #             client_data[client_id]['b_share'],
                    #             client_data[client_id]['seed_list_share']
                    #         )
                    # else:
                    if client_id == logical_client_id:  # logical_client_id can only go here
                        secret_dict[client_id] = self_b_share
                    else:
                        secret_dict[client_id] = client_data[client_id]['b_share']
                else:
                    secret_dict[client_id] = client_data[client_id]['s_sk_share']

            response["payload"]["info"]["secret_dict"] = secret_dict

        return response

    def clean_a_chunk(self, round_idx, chunk_idx):
        self.batch_delete_shared_values(keys=[
            'b', 'agg_res', 'data', 'b_share', 's_sk_share',
            'client_data', 'c_keypair', 's_keypair', 'u_2',
            'involved_clients'
        ],
            postfix=[round_idx, chunk_idx])
        if hasattr(Config().agg, "differential_privacy"):
            self.delete_a_shared_value(
                key=["sample_hadamard_seed", round_idx, chunk_idx]
            )
            if hasattr(Config().agg.differential_privacy,
                       "dropout_resilience") \
                    and (Config().agg.differential_privacy
                                 .dropout_resilience.type == "simple"
                         or Config().agg.differential_privacy
                                 .dropout_resilience.type == "log2"):
                self.delete_a_shared_value(
                    key=["excessive_noise_seeds", round_idx, chunk_idx]
                )
        gc.collect()