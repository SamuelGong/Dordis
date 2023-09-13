import copy
import os
import time
import logging
import numpy as np
import multiprocessing as mp
from abc import abstractmethod, ABC
from dordis.config import Config
from dordis.utils import multiprocess_handler
from dordis.utils.quantizer import quantize
from dordis.utils.batcher import batch
from dordis.utils.misc import get_chunks_idx
from dordis.utils.share_memory_handler import \
    TO_PUBLISH_SEND_TASK, NUM_SAMPLED_CLIENTS, CLIENT_ATTENDANCE, BREAK_SCHEDULER_LOOP
from dordis.utils.trace_related import find_surviving
from dordis.client_samplers.base import ClientSamplePlugin
from dordis.clients.delay_mocking_plugin import DelayMockingPlugin


class ChunkMetaPluginBase(ABC):
    def __init__(self):
        super(ChunkMetaPluginBase, self).__init__()
        self.chunk_size = {}
        self.num_chunks = None

    @abstractmethod
    def padding_dim(self, dim):
        """ """

    @abstractmethod
    def calc_chunk_size(self, dim, calc_dp_params=True):
        """ """

    @abstractmethod
    def post_calc_chunk_size(self, chunk_size, calc_dp_params=True):
        """ """


class Protocol(multiprocess_handler.CustomizedMPBase,
               ChunkMetaPluginBase):
    def __init__(self, client_id):
        multiprocess_handler.CustomizedMPBase\
            .__init__(self, client_id=client_id)
        ChunkMetaPluginBase.__init__(self)
        self.total_clients = Config().clients.total_clients
        self.client_sampling_rate_upperbound = None
        self.num_sampled_clients_upperbound = None

    def padding_dim(self, dim):
        return dim  # no padding

    def post_calc_chunk_size(self, chunk_size, calc_dp_params=True):
        pass

    # # first pad then divide
    # def calc_chunk_size(self, original_dim, calc_dp_params=True):
    #     dim = self.padding_dim(original_dim)
    #     chunk_size = {}
    #     if hasattr(Config().agg, "pipeline") \
    #             and Config().agg.pipeline.type == "even":  # even
    #         self.num_chunks = Config().agg.pipeline.num_chunks \
    #             if hasattr(Config().agg.pipeline, "num_chunks") \
    #             else 1
    #
    #         s = list(get_chunks_idx(dim, self.num_chunks))
    #         for chunk_idx in range(self.num_chunks):
    #             begin, end = s[chunk_idx]
    #             chunk_size[chunk_idx] = end - begin
    #     else:
    #         self.num_chunks = 1
    #         chunk_size[0] = dim
    #
    #     self.chunk_size = chunk_size
    #     logging.info(f"Chunk size calculated for dim {original_dim} "
    #                  f"(padded to {dim}): {chunk_size}.")
    #
    #     self.post_calc_chunk_size(chunk_size, calc_dp_params=calc_dp_params)
    #     return chunk_size

    # first divide then pad
    def calc_chunk_size(self, original_dim, calc_dp_params=True):
        chunk_size = {}
        if hasattr(Config().agg, "pipeline") \
                and Config().agg.pipeline.type == "even":  # even
            self.num_chunks = Config().agg.pipeline.num_chunks \
                if hasattr(Config().agg.pipeline, "num_chunks") \
                else 1

            s = list(get_chunks_idx(original_dim, self.num_chunks))
            for chunk_idx in range(self.num_chunks):
                begin, end = s[chunk_idx]
                chunk_size[chunk_idx] = end - begin
        else:
            self.num_chunks = 1
            chunk_size[0] = original_dim

        self.chunk_size = chunk_size
        logging.info(f"Chunk size calculated for dim {original_dim} : {chunk_size}.")
        if hasattr(Config().agg, "differential_privacy"):
            padded_chunk_size = {}
            total = 0
            for chunk_idx, s in chunk_size.items():
                padded_dim = self.padding_dim(s)
                padded_chunk_size[chunk_idx] = padded_dim
                total += padded_dim
            logging.info(f"To be padded for DP: dim {total}: {padded_chunk_size}.")

        self.post_calc_chunk_size(chunk_size, calc_dp_params=calc_dp_params)
        return chunk_size

    def set_client_sampling_rate_upperbound(self, rate):  # for DP accounting
        self.client_sampling_rate_upperbound = rate

    def set_num_sampled_clients_upperbound(self, num):  # for DP accounting, too
        self.num_sampled_clients_upperbound = num

    def get_record_key_for_a_phase(self, round_idx, chunk_idx, phase_idx):
        return self._keys_of_a_prefix(prefix=[
            'record', round_idx, chunk_idx, phase_idx
        ])

    def get_dropout_record_key_for_a_chunk(self, round_idx, chunk_idx):
        return self._keys_of_a_prefix(prefix=[
            'dropout_record', round_idx, chunk_idx
        ])

    def handler_head(self, aux):
        self.set_log()
        key = self.get_process_key(aux)

        self.set_a_status(
            key=key,
            value={"pid": os.getpid()}
        )

    def handler_tail(self, aux, response):
        key = self.get_process_key(aux)
        status = self.get_a_status(key=key)

        if isinstance(response, dict) and response:  # so that empty dict can skip
            response = [response]
        if isinstance(response, list):
            for _idx, resp in enumerate(response):
                if resp and isinstance(resp, dict):
                    log_prefix_str = ""
                    if "log_prefix_str" in resp:
                        log_prefix_str = resp["log_prefix_str"]

                    prompt = ""
                    if "prompt" in resp:
                        prompt = resp["prompt"]

                    # schedule sending
                    if "payload" in resp:
                        payload = resp["payload"]

                        send_list = None
                        if "send_list" in resp:
                            send_list = resp["send_list"]

                        send_list_round_idx = None
                        if "send_list_round_idx" in resp:
                            send_list_round_idx = resp["send_list_round_idx"]

                        key_postfix = None  # TODO
                        if "key_postfix" in resp:
                            key_postfix = resp["key_postfix"]
                        if key_postfix is not None:
                            if not isinstance(key_postfix, list):
                                key_postfix = [key_postfix]
                            key_postfix += [_idx]
                        else:
                            key_postfix = [_idx]

                        # mainly for the server
                        send_type = "default"
                        if "send_type" in resp:
                            send_type = resp["send_type"]

                        message = {
                            'type': send_type,
                            'list': send_list,
                            'list_round_idx': send_list_round_idx,
                            'payload': payload,
                            'key_postfix': key_postfix,
                            'log_prefix_str': log_prefix_str,
                            'prompt': prompt  # prompt after sending
                        }
                        self.send_message(
                            message=message,
                            send_list=send_list,
                            key_postfix=key_postfix,  # for PORT_SEND
                            key=key,  # for TO_PUBLISH_SEND_TASK
                            resp_idx=_idx
                        )
                    else:  # prompt now
                        if prompt:
                            log_content = f"{log_prefix_str} {prompt}".strip()
                            logging.info(log_content)

        status["done"] = True
        self.delete_a_status(key=key)

    @abstractmethod
    def send_message(self, message, send_list, key_postfix, key, resp_idx):
        """ """

    def quantize_data(self, data, quantization_params, log_prefix_str,
                      for_addition=True, padded_num_bits_to_subtract=None):
        logging.info(f"[Debug] Before quantization: {[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]} "
                     f"max: {max(data)}, min: {min(data)}.")

        if for_addition: # need to subtract padded_num_bits
            assert padded_num_bits_to_subtract is not None
            data = quantize(
                flatten_array=data,
                params=quantization_params,
                padded_num_bits=padded_num_bits_to_subtract
            )
        else:
            data = quantize(
                flatten_array=data,
                params=quantization_params,
            )

        logging.info(f"[Debug] After quantization: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")
        logging.info("%s Data quantized.", log_prefix_str)
        return data

    def batch_data(self, data, batching_params, bits_per_element, log_prefix_str):
        ''' batching is only for more precise control of traffic '''
        logging.info(f"[Debug] Before batching: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")

        if batching_params.type == "best":
            total_bit_width = batching_params.total_bit_width
            batch_size = total_bit_width // bits_per_element
        else:  # type == "fixed"
            batch_size = batching_params.batch_size
            total_bit_width = batch_size * bits_per_element

        data = batch(
            flatten_array=data,
            bits_per_element=bits_per_element,
            batch_size=batch_size
        )
        logging.info(f"[Debug] After batching: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")
        logging.info("%s Data batched (bits_per_element=%d, "
                     "batch_size=%d, total_bit_width=%d).",
                     log_prefix_str, bits_per_element,
                     batch_size, total_bit_width)
        return data

    def unquantize_data(self, data, quantization_params, log_prefix_str,
                        for_addition=True, num_involve_clients=None,
                        padded_num_bits_to_subtract=None,
                        aux=None):  # for dp and dps
        logging.info(f"[Debug] Before undequantization: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")

        # is_fl_app = Config().app.type == "federated_learning"
        if not for_addition:
            data = quantize(
                flatten_array=data,
                params=quantization_params,
                l=1,
            )
        else:  # for addition is True
            assert padded_num_bits_to_subtract is not None
            assert num_involve_clients is not None
            data = quantize(
                flatten_array=data,
                params=quantization_params,
                l=num_involve_clients,
                padded_num_bits=padded_num_bits_to_subtract
            )
        logging.info(f"[Debug] After unquantization: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")
        logging.info("%s Data unquantized.", log_prefix_str)
        return data

    def unbatch_data(self, data, batching_params, bits_per_element,
                     original_length, log_prefix_str):
        logging.info(f"[Debug] Before unbatching: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")

        if batching_params.type == "best":
            total_bit_width = batching_params.total_bit_width
            batch_size = total_bit_width // bits_per_element
        else:  # type == "fixed"
            batch_size = batching_params.batch_size
            total_bit_width = batch_size * bits_per_element

        data = batch(
            flatten_array=data,
            bits_per_element=bits_per_element,
            batch_size=batch_size,
            original_length=original_length
        )

        logging.info(f"[Debug] After unbatching: "
                     f"{[round(e, 4) for e in data[:3]]} "
                     f"{[round(e, 4) for e in data[-3:]]}.")
        logging.info("%s Data unbatched (bits_per_element=%d, "
                     "batch_size=%d, total_bit_width=%d).",
                     log_prefix_str, bits_per_element,
                     batch_size, total_bit_width)
        return data

    @abstractmethod
    def clean_a_chunk(self, round_idx, chunk_idx):
        """ """


class ProtocolServer(Protocol, ClientSamplePlugin, ABC):
    def __init__(self, client_id):
        Protocol.__init__(self, client_id=client_id)
        ClientSamplePlugin.__init__(self, client_id=client_id)
        self.graph_dict = {}
        self.threshold_dict = {}
        self.wait_time = Config().agg.wait_time \
            if hasattr(Config().agg, "wait_time") else 3600
        self.set_graph_dict()

        self.dropout_mocking_phase = None
        self.dropout_mocking_plan = {}
        self.client_sampler = None

    def set_client_sampler(self, client_sampler):
        self.client_sampler = client_sampler

    def set_client_manager(self, client_mamager):
        self.client_manager = client_mamager

    def record_attendance_and_sample_next_round(self, client_list, round_idx,
                                                log_prefix_str, abort_message_list):
        # record attendance
        if hasattr(Config().clients, "attending_rate_upperbound") \
                or hasattr(Config().clients, "attending_time_upperbound"):
            attendance_record = self.get_a_shared_value(
                key=CLIENT_ATTENDANCE
            )
            for client_id in client_list:
                attendance_record[client_id] += 1
            self.set_a_shared_value(
                key=CLIENT_ATTENDANCE,
                value=attendance_record
            )

            logging.info(f"{log_prefix_str} Attendance updated to {attendance_record}.")

        # And sample for next round
        available_clients = self.client_manager.get_available_clients()
        logging.info(f"{log_prefix_str} [Round {round_idx}] "
                     f"Available clients: {available_clients}.")

        self.client_sampler.sample(  # sampling in advance is mainly for FL
            candidates=available_clients,  # TODO: to aware of dropout
            round_idx=round_idx + 1,
            log_prefix_str=log_prefix_str
        )
        sampled_clients_next_round = self.fast_get_sampled_clients(round_idx=round_idx+1)
        if sampled_clients_next_round is None:
            logging.info(f"{log_prefix_str} Aborting due to no clients "
                         f"sampled in the next round.")
            for abort_message in abort_message_list:
                logging.info(abort_message)
            self._publish_a_value(
                channel=[BREAK_SCHEDULER_LOOP],
                message=1  # placeholder
            )
            # to avoid proceeding too much and the leaking semaphore
            time.sleep(3)

    def handler_tail(self, aux, response):
        # instrumenting for macking dropout
        if self.dropout_mocking_phase is not None:
            if isinstance(response, dict) and response:
                # currently only phases that have interactions can
                # the dropout instruction be piggybacked
                # however, that is enough

                if "send_type" in response \
                        and response["send_type"] == "exchange":
                    send_list = sorted(list(response["payload"].keys()))
                    payload_for_the_first_client = response["payload"][send_list[0]]
                    phase_idx = payload_for_the_first_client["phase"]

                    if phase_idx == self.dropout_mocking_phase:
                        round_idx = payload_for_the_first_client["round"]
                        dropout_mocking_plan \
                            = self.get_dropout_mocking_plan(round_idx)
                        clients_to_drop = [client_id for client_id in send_list
                                           if client_id in dropout_mocking_plan]

                        for dst_client_id, payload in response["payload"].items():
                            if dst_client_id in clients_to_drop:
                                payload.update({
                                    'dropout': True
                                })
                                response["payload"][dst_client_id] = payload
                else:  # broadcast
                    phase_idx = response["payload"]["phase"]

                    if phase_idx == self.dropout_mocking_phase:
                        round_idx = response["payload"]["round"]
                        send_list = response["send_list"]

                        dropout_mocking_plan \
                            = self.get_dropout_mocking_plan(round_idx)
                        clients_to_drop = [client_id for client_id in send_list
                                           if client_id in dropout_mocking_plan]
                        clients_to_stay = [client_id for client_id in send_list
                                           if client_id not in dropout_mocking_plan]

                        instrumented_response = []
                        response_for_clients_to_stay = copy.deepcopy(response)
                        response_for_clients_to_stay["send_list"] = clients_to_stay
                        instrumented_response.append(response_for_clients_to_stay)

                        response_for_clients_to_drop = copy.deepcopy(response)
                        if "payload" not in response_for_clients_to_drop:
                            response_for_clients_to_drop["payload"] = {}
                        response_for_clients_to_drop["payload"]["dropout"] = True
                        response_for_clients_to_drop["send_list"] = clients_to_drop
                        instrumented_response.append(response_for_clients_to_drop)

                        response = instrumented_response

        super().handler_tail(aux, response)

    def gradually_publish_a_value(self, channel, message, send_list, key_postfix):
        num_cpus_at_a_time = mp.cpu_count() // 4  # TODO: avoid hard-coding

        # remember that we have two types of sending
        send_type = message["type"]
        if send_type == "exchange":
            send_list = list(message["payload"].keys())

        num_batches = (len(send_list) - 1) // num_cpus_at_a_time + 1
        send_list_list = [send_list[begin:end]
                          for begin, end in get_chunks_idx(len(send_list), num_batches)]
        send_list_list = [e for e in send_list_list if len(e) > 0]
        # note that round_idx may not equate _round_idx
        _round_idx, chunk_idx, phase_idx = key_postfix[0:3]

        accumulated_sum = 0
        expected_sum = 0
        sleep_interval = 1  # TODO: avoid hard-coding

        for batch_idx, send_list in enumerate(send_list_list):
            # send
            actual_message = copy.deepcopy(message)

            if send_type == "exchange":
                actual_message["payload"] = {}
                for client_id in send_list:
                    actual_message["payload"][client_id] = message["payload"][client_id]

                # no need to bother with message["list"] as it is None
                # and we do not use message["list"] in exchange
            else:  # "default" for broadcast
                actual_message["list"] = send_list

            # important! otherwise some unsent Redis content can be occasionally removed
            # E.g., (from the perspective of io_executing) original: port_send_8003/0/0/5/0/body (no batch_idx)
            # change to : port_send_8003/0/0/5/0/(batch_idx)/body
            # codesign with server/base.py: func: sending_and_cleaning
            actual_message["key_postfix"].append(batch_idx)

            self._publish_a_value(
                channel=channel + [batch_idx],  # need to append this as otherwise Redis will be cleared
                message=actual_message,
                subscriber_only_knows_prefix=True,
                mode="large"
            )
            # logging.info(f"Gradually {_round_idx}/{chunk_idx}/{phase_idx}: {channel + [batch_idx]}.")
            if batch_idx == len(send_list_list) - 1:
                break

            expected_sum += len(send_list)
            while accumulated_sum < expected_sum:
                time.sleep(sleep_interval)
                not_dropped_key = self.get_record_key_for_a_phase(
                    round_idx=_round_idx,
                    chunk_idx=chunk_idx,
                    phase_idx=phase_idx,
                )
                dropped_key = self.get_dropout_record_key_for_a_phase(
                    round_idx=_round_idx,
                    chunk_idx=chunk_idx,
                    phase_idx=phase_idx
                )

                # as we are in simulation mode we do not need to
                # worry about the incident of client disconnecting
                accumulated_sum = len(not_dropped_key) + len(dropped_key)
                # logging.info(f"[Debug] Acc {accumulated_sum} = {len(not_dropped_key)} + {len(dropped_key)}. "
                #              f"Exp {expected_sum}, _round/chunk/"
                #              f"phase {_round_idx}/{chunk_idx}/{phase_idx} _idx: {_idx}.")

    def send_message(self, message, send_list, key_postfix, key, resp_idx):
        # logging.info(f"[Debug] send_message {[TO_PUBLISH_SEND_TASK, key, resp_idx]} to {send_list}.")
        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            # for reducing resource contention at the node for the simulation
            # the logic may be a little bit weird, but it is supposed not to
            # impact the functionality
            self.gradually_publish_a_value(
                channel=[TO_PUBLISH_SEND_TASK, key, resp_idx],
                message=message,
                send_list=send_list,
                key_postfix=key_postfix
            )
        else:
            self._publish_a_value(
                channel=[TO_PUBLISH_SEND_TASK, key, resp_idx],
                message=message,
                subscriber_only_knows_prefix=True,
                mode="large"
            )

    def get_record_for_a_phase(self, round_idx, chunk_idx, phase_idx):
        return self.prefix_to_dict([
            'record', round_idx, chunk_idx, phase_idx
        ])

    def get_records_for_phases(self, round_idx, chunk_idx, phases):
        result = []
        for phase_idx in phases:
            result.append(self.get_record_for_a_phase(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=phase_idx
            ))
        return result

    def delete_record_for_a_phase(self, round_idx, chunk_idx, phase_idx):
        self.delete_a_prefix([
            'record', round_idx, chunk_idx, phase_idx
        ])

    def delete_records_for_phases(self, round_idx, chunk_idx, phases):
        for phase_idx in phases:
            self.delete_record_for_a_phase(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=phase_idx
            )

    def get_graph_dict(self):
        return self.graph_dict

    def execute_a_task(self, task_info):
        round_idx, chunk_idx, phase_idx = task_info
        self.spawn_to_handle(
            aux=('active', round_idx, chunk_idx, phase_idx),
            routine=self.graph_dict[phase_idx]["worker"],
            args=(round_idx, chunk_idx)
        )

    def wait_for_possible_clients(self, round_idx, chunk_idx,
                                  phase_idx, start_time):
        # TODO: currently can allow >, consider making it nicer :)
        # because in FL, in some phases, the dropped_out_key
        # corresponds to different round with that which keys corresponds to

        if self.test_next_round_threshold(phase_idx):
            num_sampled_clients \
                = self.get_num_sampled_clients(round_idx=round_idx + 1)
        else:
            num_sampled_clients \
                = self.get_num_sampled_clients(round_idx=round_idx)

        while time.perf_counter() - start_time < self.wait_time:
            keys = self.get_record_key_for_a_phase(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=phase_idx
            )
            dropped_out_key = self.get_dropout_record_key_for_a_phase(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=phase_idx
            )

            # logging.info(f"[Debug] In waiting: {len(keys)}:{keys}, "
            #              f"{len(dropped_out_key)}:{dropped_out_key}, "
            #              f"{num_sampled_clients}")
            if len(keys) + len(dropped_out_key) >= num_sampled_clients:
                break
            time.sleep(0.001)
        self.threshold_test_pass(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )

    @abstractmethod
    def threshold_test_pass(self, round_idx, chunk_idx, phase_idx):
        """ """

    def get_dropout_record_key_for_a_phase(self, round_idx, chunk_idx, phase_idx):
        if phase_idx >= self.DOWNLOAD_DATA:
            # TODO: this is application-specific (e.g., FL)
            # avoid hard-coding
            return self.get_dropout_record_key_for_a_chunk(
                round_idx=round_idx + 1,
                chunk_idx=chunk_idx
            )
        else:
            return self.get_dropout_record_key_for_a_chunk(
                round_idx=round_idx,
                chunk_idx=chunk_idx
            )

    @abstractmethod
    def test_next_round_threshold(self, phase_idx):
        """" """

    def threshold_test(self, round_idx, chunk_idx, phase_idx):
        keys = self.get_record_key_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )
        if not keys:
            return False

        dropped_out_key = self.get_dropout_record_key_for_a_phase(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx
        )

        cnt = len(keys)
        if cnt < self.get_threshold(round_idx=round_idx):
            return False
        else:
            waited = self.get_a_shared_value(
                key=['waited', round_idx, chunk_idx, phase_idx]
            )
            if waited is not None:  # has already been awaited
                return True
            else:
                waiting = self.get_a_shared_value(
                    key=['waiting', round_idx, chunk_idx, phase_idx]
                )
                if waiting is not None:  # is being waited by another process
                    return False

                self.set_a_shared_value(
                    key=['waiting', round_idx, chunk_idx, phase_idx],
                    value=True,
                )

                # e.g., DOWNLOAD uses a new set of clients
                if self.test_next_round_threshold(phase_idx):
                    num_sampled_clients \
                        = self.get_num_sampled_clients(round_idx=round_idx + 1)
                else:
                    num_sampled_clients \
                        = self.get_num_sampled_clients(round_idx=round_idx)

                if cnt + len(dropped_out_key) < num_sampled_clients:
                    # logging.info(f"[Debug] to wait: cnt={cnt} ({keys}), "
                    #              f"dropped {len(dropped_out_key)} ({dropped_out_key}),"
                    #              f"num_sampled_clients: {num_sampled_clients}.")
                    start_time = time.perf_counter()
                    p = mp.Process(
                        target=self.wait_for_possible_clients,
                        args=(round_idx, chunk_idx,
                              phase_idx, start_time)
                    )
                    p.start()
                else:
                    # logging.info(f"[Debug] no need to wait: cnt={cnt} ({keys}), "
                    #              f"dropped {len(dropped_out_key)} ({dropped_out_key}),"
                    #              f"num_sampled_clients: {num_sampled_clients}.")
                    self.threshold_test_pass(
                        round_idx=round_idx,
                        chunk_idx=chunk_idx,
                        phase_idx=phase_idx
                    )
                return True

    def handle_client_payload(self, client_id, payload, payload_size):
        """ Should be a stateless handler due to multiprocessing. """
        if isinstance(payload, dict):
            round_idx = payload['round']
            chunk_idx = payload['chunk']
            logical_client_id = payload['logical_client_id']
            completed_chunks = self.get_a_shared_value(
                key=['completed', round_idx, chunk_idx]
            )

            if completed_chunks is not None:
                log_prefix_str = self.get_log_prefix_str()
                logging.info(
                    "%s Received %s MB of payload data from client #%d. "
                    "Discarding it as round %d has completed.", log_prefix_str,
                    round(payload_size / 1024 ** 2, 6),
                    logical_client_id, round_idx)
                return

            phase_idx = payload['phase']
            log_prefix_str = self.get_log_prefix_str(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=phase_idx
            )

            if self.dropout_mocking_phase is not None \
                    and "drop_out" in payload:
                logging.info("%s client #%d just dropped out.",
                    log_prefix_str, logical_client_id)
                self.set_a_shared_value(
                    key=['dropout_record', round_idx, chunk_idx, logical_client_id],
                    value=1  # placeholder
                )

                self.threshold_test(
                    round_idx=round_idx,
                    chunk_idx=chunk_idx,
                    phase_idx=phase_idx
                )
            else:
                logging.info(
                    "%s Received %s MB of payload data from client #%d.",
                    log_prefix_str, round(payload_size / 1024 ** 2, 6),
                    logical_client_id)

                self.store_client_payload(
                    args=(logical_client_id, payload, round_idx, chunk_idx, phase_idx)
                )
                logging.info("%s Stored payload data from client #%d.",
                    log_prefix_str, logical_client_id)
        else:
            log_prefix_str = self.get_log_prefix_str()
            logging.info(
                "%s Received %s MB of payload data from physical client #%d. "
                "Discarding it for being irrelevant to the protocol.",
                log_prefix_str, round(payload_size / 1024 ** 2, 6), client_id)

    def get_dropout_mocking_plan(self, round_idx):
        if round_idx not in self.dropout_mocking_plan:
            type = Config().clients.dropout.type
            sampled_clients = self.fast_get_sampled_clients(round_idx)

            if type == "fixed_frac":
                num_sampled_clients \
                    = self.get_num_sampled_clients(round_idx=round_idx)
                frac = Config().clients.dropout.args[0]
                num_choice = int(frac * num_sampled_clients)
                np.random.seed(round_idx)  # to be consistent across chunks
                self.dropout_mocking_plan[round_idx] \
                    = np.random.choice(sampled_clients, num_choice,
                                       replace=False).tolist()
            elif type == "trace_driven":
                trace_related = self.get_trace_related(round_idx=round_idx)
                survivals = trace_related["survivals"]  # precomputed
                if hasattr(Config(), "simulation"):  # currently only support simulation
                    dropout_clients = [e for e in sampled_clients if e not in survivals]
                    self.dropout_mocking_plan[round_idx] = dropout_clients
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        clients_to_drop = self.dropout_mocking_plan[round_idx]
        logging.info(f"Mocking client dropout: {clients_to_drop}.")
        return clients_to_drop

    @abstractmethod
    def set_graph_dict(self):
        """ """

    @abstractmethod
    def get_threshold(self, round_idx):
        """ """

    @abstractmethod
    def store_client_payload(self, args):
        """ """


class ProtocolClient(Protocol, DelayMockingPlugin, ABC):
    def __init__(self, client_id):
        Protocol.__init__(self, client_id)
        DelayMockingPlugin.__init__(self, client_id)
        self.num_sampled_clients = {}
        self.routine = {}
        self.set_routine()

    def set_num_sampled_clients(self, round_idx, num_sampled_clients):
        self.num_sampled_clients[round_idx] = num_sampled_clients
        self.set_a_shared_value(
            key=[NUM_SAMPLED_CLIENTS, round_idx],
            value=num_sampled_clients
        )

    def get_num_sampled_clients(self, round_idx):
        if round_idx not in self.num_sampled_clients:
            num_sampled_clients = self.get_a_shared_value(
                key=[NUM_SAMPLED_CLIENTS, round_idx]
            )
            self.num_sampled_clients[round_idx] = num_sampled_clients

        return self.num_sampled_clients[round_idx]

    def send_message(self, message, send_list, key_postfix, key, resp_idx):
        self._publish_a_value(
            channel=[TO_PUBLISH_SEND_TASK, key, resp_idx],
            message=message,
            subscriber_only_knows_prefix=True,
            mode="large"
        )

    def handle_server_payload(self, payload, payload_size):
        if isinstance(payload, dict):
            round_idx = payload['round']
            chunk_idx = payload['chunk']
            phase_idx = payload['phase']
            logical_client_id = payload['logical_client_id']

            log_prefix_str = self.get_log_prefix_str(
                round_idx=round_idx,
                chunk_idx=chunk_idx,
                phase_idx=phase_idx,
                logical_client_id=logical_client_id
            )
            logging.info(
                "%s Received %s MB of payload data from the server.",
                log_prefix_str, round(payload_size / 1024 ** 2, 6))

            if "dropout" in payload and payload["dropout"]:  # mocking dropout
                logging.info("%s Mocking dropout at %s.",
                             log_prefix_str, self.routine[phase_idx])
                self.spawn_to_handle(
                    aux=('passive', round_idx, chunk_idx, phase_idx),
                    routine="drop_out",
                    args=(round_idx, chunk_idx, phase_idx, logical_client_id),
                    delay_mocking_factor=0
                )
            else:
                logging.info("%s Received to run %s.",
                         log_prefix_str, self.routine[phase_idx])

                delay_mocking_factor = 0
                if hasattr(Config().clients, "delay_mock"):
                    delay_mocking_factor \
                        = self.fast_get_delay_mocking_factors(logical_client_id)

                self.spawn_to_handle(
                    aux=('passive', round_idx, chunk_idx, phase_idx),
                    routine=self.routine[phase_idx],
                    args=(payload, round_idx, chunk_idx,
                          phase_idx, logical_client_id),
                    delay_mocking_factor=delay_mocking_factor
                )
        else:  # traffic out of the protocol
            log_prefix_str = self.get_log_prefix_str()
            logging.info(
                "%s Received %s MB of payload data from the server.",
                log_prefix_str, round(payload_size / 1024 ** 2, 6))

    def drop_out(self, args):
        round_idx, chunk_idx, phase_idx, logical_client_id = args
        log_prefix_str = self.get_log_prefix_str(
            round_idx=round_idx,
            chunk_idx=chunk_idx,
            phase_idx=phase_idx,
            logical_client_id=logical_client_id
        )
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
                'drop_out': True,
                'logical_client_id': logical_client_id
            },
            "key_postfix": [round_idx, chunk_idx, phase_idx],
            "log_prefix_str": log_prefix_str,
            "prompt": "Dropout mocked."
        }

    @abstractmethod
    def set_routine(self):
        """ """
