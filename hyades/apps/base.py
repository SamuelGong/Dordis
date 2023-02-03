import threading
import os
import time
import redis
import logging
import pickle
import signal
from hyades.config import Config
from hyades.utils.misc import calc_sleep_time, mocking_cpu_intensive
from abc import abstractmethod
from hyades.utils.multiprocess_handler import CustomizedMPBase
from hyades.utils.share_memory_handler import redis_pool, \
    AGG_RES_PREPARED_FOR_SERVER, CLOSE_SERVER, \
    CLOSE_CLIENT, TO_PREPARE_DATA, AGG_RES_PREPARED_FOR_CLIENT

r = redis.Redis(connection_pool=redis_pool)


class App(CustomizedMPBase):
    def __init__(self, client_id):
        super(App, self).__init__(client_id=client_id)
        self.total_clients = Config().clients.total_clients
        self.debug = False
        self.to_stop = False
        # avoid conflict with what "protocol" uses
        self.set_status_prefix(prefix="app-status")

    def handler_head(self, aux):
        self.set_log()
        key = self.get_process_key(aux)

        self.set_a_status(
            key=key,
            value={
                "done": False,
                "pid": os.getpid()
            }
        )

    def handler_tail(self, aux, response):
        key = self.get_process_key(aux)
        status = self.get_a_status(key=key)

        status["done"] = True
        self.set_a_status(
            key=key,
            value=status
        )

    def cleaning_loop(self):
        while True:
            exit_after_traverse = False
            if self.to_stop:
                exit_after_traverse = True

                status_dict = self.get_status_dict()
                for key, status in status_dict.items():
                    if status["done"] or exit_after_traverse:
                        pid = status["pid"]
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # the program has already dead
                        self.delete_a_status(key=key)

            if exit_after_traverse:
                break

    @abstractmethod
    def use_output(self, args):
        """ It should publish to the channel
        AGG_RES_USED_BY_SERVER or AGG_RES_USED_BY_CLIENT. """


class AppServer(App):
    def __init__(self, client_id=0):
        super().__init__(client_id=0)
        if hasattr(Config().app, "debug") \
                and hasattr(Config().app.debug, "server"):
            self.debug = True

    def run(self):
        t = threading.Thread(target=self.running_loop)
        t.start()

        t = threading.Thread(target=self.cleaning_loop)
        t.start()

    def running_loop(self):
        sub, ch_dict = self.batch_subscribe_channels(d={
            AGG_RES_PREPARED_FOR_SERVER: True,
            CLOSE_SERVER: False
        })
        agg_res_prepared_ch = ch_dict[AGG_RES_PREPARED_FOR_SERVER]

        for message in sub.listen():
            if self.to_stop:
                break
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue
            channel = message['channel'].decode()
            channel = self.strip_self_channel_prefix(channel)

            if channel == CLOSE_SERVER:
                self.to_stop = True
                break
            elif channel == agg_res_prepared_ch:
                _data = pickle.loads(raw_data)
                key = _data['key']
                data = self.get_a_shared_value(key=key)
                self.delete_a_shared_value(key=key)

                agg_res = data["agg_res"]
                involved_clients = data["involved_clients"]
                round_idx = data["round_idx"]
                chunk_idx = data["chunk_idx"]
                log_prefix_str = data["log_prefix_str"]

                self.spawn_to_handle(
                    aux=(agg_res_prepared_ch, round_idx, chunk_idx),
                    routine="use_output",
                    args=(round_idx, chunk_idx, log_prefix_str, agg_res,
                          involved_clients),
                )


class AppClient(App):
    def __init__(self, client_id):
        super().__init__(client_id=client_id)
        if hasattr(Config().app, "debug") \
                and hasattr(Config().app.debug, "client"):
            self.debug = True

    def run(self):
        t = threading.Thread(target=self.running_loop)
        t.start()

        t = threading.Thread(target=self.cleaning_loop)
        t.start()

    def running_loop(self):
        sub, ch_dict = self.batch_subscribe_channels(d={
            CLOSE_CLIENT: False,
            TO_PREPARE_DATA: False,
            AGG_RES_PREPARED_FOR_CLIENT: True
        })
        close_ch = ch_dict[CLOSE_CLIENT]
        to_prepare_data_ch = ch_dict[TO_PREPARE_DATA]
        agg_res_prepare_ch = ch_dict[AGG_RES_PREPARED_FOR_CLIENT]

        for message in sub.listen():
            if self.to_stop:
                break
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue
            channel = message['channel'].decode()
            channel = self.strip_self_channel_prefix(channel)

            if channel == close_ch:
                self.to_stop = True
                break
            elif channel == to_prepare_data_ch:
                data = pickle.loads(raw_data)
                round_idx = data["round_idx"]
                chunk_idx = data["chunk_idx"]
                logical_client_id = data["logical_client_id"]
                log_prefix_str = data["log_prefix_str"]
                self.spawn_to_handle(
                    aux=(to_prepare_data_ch,
                         round_idx, chunk_idx),
                    routine="prepare_data",
                    args=(round_idx, chunk_idx, log_prefix_str, logical_client_id),
                )
            elif channel == agg_res_prepare_ch:
                _data = pickle.loads(raw_data)
                key = _data['key']
                data = self.get_a_shared_value(key=key)
                self.delete_a_shared_value(key=key)

                round_idx = data["round_idx"]
                chunk_idx = data["chunk_idx"]
                logical_client_id = data["logical_client_id"]
                log_prefix_str = data["log_prefix_str"]
                agg_res = data['agg_res']
                involved_clients = data['involved_clients']
                self.spawn_to_handle(
                    aux=(agg_res_prepare_ch, round_idx, chunk_idx),
                    routine="use_output",
                    args=(round_idx, chunk_idx, log_prefix_str, agg_res,
                          involved_clients, logical_client_id),
                )

    def mocking_preparation_time(self, start_time, num_elements, log_prefix_str):
        """ Mock the latency of data preparation. """
        if hasattr(Config().app, "prepare_data_time_mock"):
            actual_duration = time.perf_counter() - start_time
            type = Config().app.prepare_data_time_mock.type
            if type == "proportionate":
                c = Config().app.prepare_data_time_mock.c
                expected_wait_time = c * num_elements
                actual_wait_time = max(0, expected_wait_time - actual_duration)

                if actual_wait_time > 0:
                    logging.info(f"{log_prefix_str} Mocking preparation latency "
                                 f"(expected wait time: {expected_wait_time}, "
                                 f"actual duration: {round(actual_duration, 2)}, "
                                 f"actual wait time: {round(actual_wait_time, 2)}).")

                    loop_interval = 0.1
                    loop_step = 0
                    start_time = time.perf_counter()
                    while True:
                        if time.perf_counter() - start_time >= actual_wait_time:
                            break
                        mocking_cpu_intensive()

                        sleep_time = calc_sleep_time(
                            sec_per_step=loop_interval,
                            cur_step=loop_step,
                            start_time=start_time
                        )
                        time.sleep(sleep_time)
                        loop_step += 1
                    logging.info(f"{log_prefix_str} Preparation latency mocked.")

    @abstractmethod
    def prepare_data(self, args):
        """ """
