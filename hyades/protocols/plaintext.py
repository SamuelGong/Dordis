import gc
import pickle
import time

import redis
import logging
import numpy as np
from hyades.config import Config
from hyades.protocols import base
from hyades.protocols.const import PlaintextConst
from hyades.utils.misc import plaintext_aggregate
from hyades.utils.share_memory_handler import \
    redis_pool, AGG_RES_USED_BY_SERVER, AGG_RES_PREPARED_FOR_SERVER, \
    DATA_PREPARED, AGG_RES_USED_BY_CLIENT, SCHEDULE, \
    TO_PREPARE_DATA, AGG_RES_PREPARED_FOR_CLIENT, \
    CLIENT_STATS, SIMULATED_CLIENT_LATENCY

r = redis.Redis(connection_pool=redis_pool)


class ProtocolServer(base.ProtocolServer, PlaintextConst):
    def __init__(self, client_id=0):
        PlaintextConst.__init__(self)
        base.ProtocolServer.__init__(self, client_id=client_id)

        if hasattr(Config().clients, "dropout"):
            if hasattr(Config().clients.dropout, "seed"):
                seed = Config().clients.dropout.seed
                np.random.seed(seed)
            self.dropout_mocking_phase = self.UPLOAD_DATA

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
            self.UPLOAD_DATA: {
                "worker": "upload_data",
                "stage": self._no_plot_phase_stage_mapping[
                    self.UPLOAD_DATA],
                "predecessor_ids": [self.ENCODE_DATA]
            },
            self.GENERATE_OUTPUT: {
                "worker": "generate_output",
                "stage": self._no_plot_phase_stage_mapping[
                    self.GENERATE_OUTPUT],
                "predecessor_ids": [self.UPLOAD_DATA],
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

    def get_threshold(self, round_idx):
        if round_idx not in self.threshold_dict:
            num_sampled_clients \
                = self.get_num_sampled_clients(round_idx=round_idx)
            threshold_frac = Config().agg.threshold \
                if hasattr(Config().agg, "threshold") else 1.0
            threshold = int(np.ceil(threshold_frac * num_sampled_clients))
            self.threshold_dict[round_idx] = threshold
        else:
            threshold = self.threshold_dict[round_idx]

        return threshold

    def threshold_test_pass(self, round_idx, chunk_idx, phase_idx):
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        self.set_a_shared_value(
            key=['waited', round_idx, chunk_idx, phase_idx],
            value=True,
        )

        dropped_out_key = self.get_dropout_record_key_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        clients_dropped_out = [int(s.split('/')[-1]) for s in dropped_out_key]

        if not clients_dropped_out:
            logging.info("%s Phase done.", log_prefix_str)
        else:
            logging.info("%s Phase done. Dropped out clients: %s",
                         log_prefix_str, clients_dropped_out)

        if phase_idx == self.CLIENT_USE_OUTPUT:
            self.clean_a_chunk(
                round_idx=round_idx,
                chunk_idx=chunk_idx
            )

        self._publish_a_value(
            channel=SCHEDULE,
            message=[round_idx, chunk_idx, phase_idx]
        )

    def store_client_payload(self, args):
        client_id, payload, round_idx, chunk_idx, phase_idx = args

        if phase_idx == self.ENCODE_DATA \
                or phase_idx == self.DOWNLOAD_DATA \
                or phase_idx == self.DECODE_DATA \
                or phase_idx == self.CLIENT_USE_OUTPUT:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=1  # placeholder
            )
        elif phase_idx == self.PREPARE_DATA:
            # logging.info(f"[A] {round_idx} {chunk_idx} {phase_idx} {payload['meta']}.")
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['meta']
            )
        elif phase_idx == self.UPLOAD_DATA:
            self.set_a_shared_value(
                key=['record', round_idx, chunk_idx, phase_idx, client_id],
                value=payload['data']
            )
        else:
            raise ValueError(f"Unknown phase: {phase_idx}.")

        self.threshold_test(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )

    def prepare_data(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.PREPARE_DATA
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        logging.info("%s Phase started. Instructing all "
                     "clients to prepare data.", log_prefix_str)

        sampled_clients = self.fast_get_sampled_clients(round_idx)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'num_sampled_clients': len(sampled_clients)
            },
            "send_list": sampled_clients,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }

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
        logging.info("%s Phase started. Instructing %d "
                     "clients to encode data: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
            },
            "send_list": surviving_clients,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }

    def upload_data(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.UPLOAD_DATA
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        encoded_data = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.ENCODE_DATA
        )
        surviving_clients = sorted(list(encoded_data.keys()))

        logging.info("%s Phase started. Instructing %d "
                     "clients to upload data: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
            },
            "send_list": surviving_clients,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }

    def update_stat(self, clients, log_prefix_str, round_idx, chunk_idx):
        stat_to_update = {}

        # Step 1: First measure clients' updated status
        # Part 1: time-related
        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple" \
                and hasattr(Config().simulation, "time"):

            for client_id in clients:
                precomputed_latency = self.get_a_shared_value(
                    key=[SIMULATED_CLIENT_LATENCY, client_id]
                )
                stat_to_update[client_id] = {
                    'time': precomputed_latency
                }
        else:  # otherwise, take each client's latency as the same
            for client_id in clients:
                stat_to_update[client_id] = {
                    'time': 1.0  # can be any other figure
                }

        # Part 2: FL-related
        if Config().app.type == "federated_learning":
            client_meta_dict = self.get_record_for_a_phase(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=self.PREPARE_DATA
            )

            for client_id in clients:
                # currently "meta" refers exactly to utility
                utility = client_meta_dict[client_id]
                if client_id in stat_to_update:
                    stat_to_update[client_id]['utility'] = utility
                else:
                    stat_to_update[client_id] = {
                        'utility': utility
                    }

        # Step 2: Update the database
        if stat_to_update:
            for client_id in clients:
                round_dict = {
                    round_idx: {
                        "used": False  # used for updating client selector
                    }
                }
                for k, v in stat_to_update[client_id].items():
                    round_dict[round_idx][k] = v

                temp = self.get_a_shared_value(
                    key=[CLIENT_STATS, client_id]
                )
                temp["round_stats"].update(round_dict)
                self.set_a_shared_value(
                    key=[CLIENT_STATS, client_id],
                    value=temp
                )

            logging.info(f"{log_prefix_str} Stats update for Round "
                         f"{round_idx}: {stat_to_update}.")

    def generate_output(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.GENERATE_OUTPUT
        client_data_dict = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.UPLOAD_DATA
        )

        surviving_clients = list(client_data_dict.keys())
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        if chunk_idx == self.num_chunks - 1:  # only update for the last chunk
            self.update_stat(
                clients=surviving_clients,
                log_prefix_str=log_prefix_str,
                round_idx=round_idx,
                chunk_idx=chunk_idx
            )
        logging.info("%s Phase started. Generating output "
                     "with %d clients: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        # unbatch if necessary before aggregation
        if hasattr(Config().agg, "quantize") \
                and hasattr(Config().agg.quantize, "batch"):
            quantize_params = Config().agg.quantize
            for k, v in client_data_dict.items():
                unbatched_v = self.unbatch_data(
                    data=v,
                    batching_params=quantize_params.batch,
                    bits_per_element=quantize_params.bit_width,
                    original_length=self.chunk_size[chunk_idx],
                    log_prefix_str=log_prefix_str
                )
                client_data_dict[k] = unbatched_v

        agg_res = plaintext_aggregate(
            data_list=list(client_data_dict.values())
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

    def server_use_output(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.SERVER_USE_OUTPUT
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        logging.info("%s Phase started. The server is using "
                     "the aggregation result.", log_prefix_str)

        agg_res, involved_clients = self.batch_get_shared_values(
            keys=['agg_res', 'involved_clients'],
            postfix=[round_idx, chunk_idx]
        )

        num_sampled_clients \
            = self.get_num_sampled_clients(round_idx=round_idx)

        if hasattr(Config().agg, "differential_privacy"):
            dp_params_dict = self.get_a_shared_value(
                key=[f'chunk{chunk_idx}_dp_params_dict', 0, chunk_idx]
            )

            # DP unquantization
            logging.info(f"[Debug] Before DP decoding: "
                         f"{[round(e, 4) for e in agg_res[:3]]} "
                         f"{[round(e, 4) for e in agg_res[-3:]]}.")
            sample_hadamard_seed = self.get_a_shared_value(
                key=["sample_hadamard_seed", round_idx, chunk_idx]
            )
            agg_res = self.dp_handler.decode_data(
                data=agg_res,
                log_prefix_str=log_prefix_str,
                other_args=(sample_hadamard_seed, dp_params_dict)
            )

            logging.info(f"[Debug] After DP unquantization: "
                         f"{[round(e, 4) for e in agg_res[:3]]} "
                         f"{[round(e, 4) for e in agg_res[-3:]]}.")
            logging.info("%s DP Unquantization done.", log_prefix_str)

            original_length = self.chunk_size[chunk_idx]
            agg_res = agg_res[:original_length]

        elif hasattr(Config().agg, "quantize"):
            quantization_params = Config().agg.quantize
            padded_num_bits = int(np.ceil(np.log2(num_sampled_clients)))
            agg_res = self.unquantize_data(
                data=agg_res,
                quantization_params=quantization_params,
                log_prefix_str=log_prefix_str,
                for_addition=True,  # for plaintext and secagg
                num_involve_clients=len(involved_clients),
                padded_num_bits_to_subtract=padded_num_bits,
                aux=(round_idx, chunk_idx),  # for dp and dps
            )

        # only when debug_agg_res or fl_aggregate_delt does agg_res make sense
        # otherwise it is just a placeholder
        self._publish_a_value(
            channel=[AGG_RES_PREPARED_FOR_SERVER, round_idx, chunk_idx],
            message={
                'agg_res': agg_res,
                'involved_clients': involved_clients,
                'round_idx': round_idx,
                'chunk_idx': chunk_idx,
                'log_prefix_str': log_prefix_str,
            },
            mode="large",
            subscriber_only_knows_prefix=True
        )

        # wait until the app has used the aggregation result
        # and return the (possibly) modified one
        sub, _ = self.subscribe_a_channel(
            channel_prefix=[AGG_RES_USED_BY_SERVER, round_idx, chunk_idx],
            is_large_value=True
        )
        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue
            break

        key = self.get_key_for_fetching_a_large_value(
            key_prefix=[AGG_RES_USED_BY_SERVER, round_idx, chunk_idx],
        )
        agg_res = self.get_a_shared_value(key=key)
        self.delete_a_shared_value(key=key)

        # record attendance of last aggregation
        # and see if we can successfully sample clients for the next round
        # before we do quantization
        idx = log_prefix_str.find("[Round")
        server_only_log_prefix_str = log_prefix_str[:idx - 1]  # strip the space
        abort_message_list = [  # if has to abort, print them for ease of analysis
            f"{log_prefix_str} Phase done.",
            f"{server_only_log_prefix_str} Round {round_idx} ended.",
        ]
        self.record_attendance_and_sample_next_round(
            client_list=involved_clients,
            round_idx=round_idx,
            log_prefix_str=log_prefix_str,
            abort_message_list=abort_message_list
        )

        # quantization for sending back to clients
        if hasattr(Config().agg, "quantize"):
            quantization_params = Config().agg.quantize
            agg_res = self.quantize_data(
                data=agg_res,
                quantization_params=quantization_params,
                log_prefix_str=log_prefix_str,
                for_addition=False
            )

            if hasattr(quantization_params, 'batch'):
                agg_res = self.batch_data(
                    data=agg_res,
                    batching_params=quantization_params.batch,
                    bits_per_element=quantization_params.bit_width,
                    log_prefix_str=log_prefix_str
                )

        self.batch_set_shared_values(d={
            'agg_res': agg_res,
            'agg_res_used': True
        }, postfix=[round_idx, chunk_idx])
        logging.info("%s Phase done.", log_prefix_str)
        self._publish_a_value(
            channel=SCHEDULE,
            message=[round_idx, chunk_idx, phase_idx]
        )
        return {}

    def test_next_round_threshold(self, phase_idx):
        if phase_idx >= self.DOWNLOAD_DATA:
            return True
        else:
            return False

    def download_data(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.DOWNLOAD_DATA

        sampled_clients_next_round = self\
            .fast_get_sampled_clients(round_idx + 1)
        num_sampled_clients_next_round = len(sampled_clients_next_round)

        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        logging.info("%s Phase started. Dispatching the "
                     "aggregation result to %d clients: %s.",
                     log_prefix_str, len(sampled_clients_next_round),
                     sampled_clients_next_round)

        agg_res, involved_clients = self.batch_get_shared_values(
            keys=['agg_res', 'involved_clients'],
            postfix=[round_idx, chunk_idx]
        )
        response = {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                "agg_res": agg_res,
                "involved_clients": involved_clients,
                'num_sampled_clients_next_round': num_sampled_clients_next_round
            },
            "send_list": sampled_clients_next_round,
            "send_list_round_idx": round_idx + 1,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
        }

        return response

    def decode_data(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.DECODE_DATA
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        downloaded_data = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.DOWNLOAD_DATA
        )
        surviving_clients = sorted(list(downloaded_data.keys()))
        logging.info("%s Phase started. Instructing %d "
                     "clients to decode data: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
            },
            "send_list": surviving_clients,
            "send_list_round_idx": round_idx + 1,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }

    def client_use_output(self, args):
        round_idx, chunk_idx = args
        phase_idx = self.CLIENT_USE_OUTPUT
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        decoded_data = self.get_record_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=self.DECODE_DATA
        )
        surviving_clients = sorted(list(decoded_data.keys()))
        logging.info("%s Phase started. Instructing %d "
                     "clients to use output: %s.", log_prefix_str,
                     len(surviving_clients), surviving_clients)

        return {
            "payload": {
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
            },
            "send_list": surviving_clients,
            "send_list_round_idx": round_idx + 1,
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str
        }

    def clean_a_chunk(self, round_idx, chunk_idx):
        self.delete_records_for_phases(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phases=[self.PREPARE_DATA, self.UPLOAD_DATA,
                    self.CLIENT_USE_OUTPUT]
        )
        self.delete_a_shared_value(key=['agg_res', round_idx, chunk_idx])
        self.clients_dropped_out = []
        gc.collect()


class ProtocolClient(base.ProtocolClient, PlaintextConst):
    def __init__(self, client_id):
        PlaintextConst.__init__(self)
        base.ProtocolClient.__init__(self, client_id)

    def set_routine(self):
        self.routine = {
            self.PREPARE_DATA: "prepare_data",
            self.ENCODE_DATA: "encode_data",
            self.UPLOAD_DATA: "upload_data",
            self.DOWNLOAD_DATA: "download_data",
            self.DECODE_DATA: "decode_data",
            self.CLIENT_USE_OUTPUT: "client_use_output"
        }

    def prepare_data(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        assert "num_sampled_clients" in payload
        self.set_num_sampled_clients(
            round_idx=round_idx,
            num_sampled_clients=payload["num_sampled_clients"]
        )

        self._publish_a_value(
            channel=TO_PREPARE_DATA,
            message={
                'round_idx': round_idx,
                'chunk_idx': chunk_idx,
                'logical_client_id': logical_client_id,
                'log_prefix_str': log_prefix_str,
            }
        )
        sub, _ = self.subscribe_a_channel(
            channel_prefix=[DATA_PREPARED, round_idx, chunk_idx],
            is_large_value=True
        )
        key = self.get_key_for_fetching_a_large_value(
            key_prefix=[DATA_PREPARED, round_idx, chunk_idx]
        )

        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue
            break
        data = self.get_a_shared_value(key=key)
        self.delete_a_shared_value(key=key)
        response = {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'logical_client_id': logical_client_id,
                'meta': 1  # placeholder, never used
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Data prepared."
        }

        if Config().app.type == "federated_learning" \
                and chunk_idx == self.num_chunks - 1:
            utility = data["utility"]
            data = data["chunk"]
            response["payload"]["meta"] = utility

        self.set_a_shared_value(
            key=['data', round_idx, chunk_idx],
            value=data
        )
        return response

    def encode_data(self, args):
        _, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        data = self.get_a_shared_value(
            key=['data', round_idx, chunk_idx]
        )

        num_sampled_clients = self.get_num_sampled_clients(round_idx=round_idx)
        if hasattr(Config().agg, "quantize"):
            quantization_params = Config().agg.quantize
            padded_num_bits = int(np.ceil(np.log2(num_sampled_clients)))
            data = self.quantize_data(
                data=data,
                quantization_params=quantization_params,
                log_prefix_str=log_prefix_str,
                padded_num_bits_to_subtract=padded_num_bits,
            )

            # batching is only possible if the data is quantized
            if hasattr(quantization_params, "batch"):
                # in SecAgg or distributed DP
                # we first apply SecAgg's mask
                # and then batch
                if Config().agg.type not in ["secagg", "dp_plus_secagg"]:  # notice this not
                    data = self.batch_data(
                        data=data,
                        batching_params=quantization_params.batch,
                        bits_per_element=quantization_params.bit_width,
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

    def upload_data(self, args):
        _, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        data = self.get_a_shared_value(
            key=['data', round_idx, chunk_idx]
        )

        return {
            "payload": {
                'client_id': self.client_id,
                'round': round_idx,
                'chunk': chunk_idx,
                'phase': phase_idx,
                'data': data,
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Data uploaded."
        }

    def download_data(self, args):
        payload, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )

        self.batch_set_shared_values(d={
            "agg_res": payload["agg_res"],
            'involved_clients': payload["involved_clients"]
        }, postfix=[round_idx, chunk_idx])

        self.set_num_sampled_clients(
            round_idx=round_idx + 1,
            num_sampled_clients=payload["num_sampled_clients_next_round"]
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
            "prompt": "Data downloaded."
        }

    def decode_data(self, args):
        _, round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
        agg_res, involved_clients = self.batch_get_shared_values(
            keys=["agg_res", "involved_clients"],
            postfix=[round_idx, chunk_idx]
        )

        if hasattr(Config().agg, "quantize"):
            quantization_params = Config().agg.quantize
            if hasattr(quantization_params, "batch"):
                agg_res = self.unbatch_data(
                    data=agg_res,
                    batching_params=quantization_params.batch,
                    bits_per_element=quantization_params.bit_width,
                    original_length=self.chunk_size[chunk_idx],
                    log_prefix_str=log_prefix_str
                )

            agg_res = self.unquantize_data(
                data=agg_res,
                quantization_params=quantization_params,
                log_prefix_str=log_prefix_str,
                for_addition=False
            )

        self.set_a_shared_value(
            key=['agg_res', round_idx, chunk_idx],
            value=agg_res
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
            "prompt": "Data decoded."
        }

    def client_use_output(self, args):
        _, round_idx, chunk_idx, phase_idx, logical_client_id = args

        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )

        agg_res, involved_clients = self.batch_get_shared_values(
            keys=["agg_res", "involved_clients"],
            postfix=[round_idx, chunk_idx]
        )

        self._publish_a_value(
            channel=[AGG_RES_PREPARED_FOR_CLIENT, round_idx, chunk_idx],
            message={
                'agg_res': agg_res,
                'logical_client_id': logical_client_id,
                'involved_clients': involved_clients,
                'round_idx': round_idx,
                'chunk_idx': chunk_idx,
                'log_prefix_str': log_prefix_str,
            },
            mode="large",
            subscriber_only_knows_prefix=True
        )

        # wait until the app has used the aggregation result
        sub, _ = self.subscribe_a_channel(
            channel_prefix=[AGG_RES_USED_BY_CLIENT, round_idx, chunk_idx]
        )

        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue
            break

        self.clean_a_chunk(
            round_idx=round_idx,
            chunk_idx=chunk_idx
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
            "prompt": "Result consumed."
        }

    def clean_a_chunk(self, round_idx, chunk_idx):
        self.batch_delete_shared_values(keys=[
            'data', 'agg_res', 'involved_clients'
        ],
            postfix=[round_idx, chunk_idx]
        )
        gc.collect()
