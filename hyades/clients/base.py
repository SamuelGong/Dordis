import os
import time

import redis
import pickle
import logging
import signal
import threading
from hyades.config import Config
from hyades.apps import registry as app_registry
from hyades.protocols import registry as protocol_registry
from hyades.client_samplers import registry as client_sampler_registry
from hyades.clients.client_proc \
    import ClientProcess, ShareBase, SEND, CLOSE_CLIENT
from hyades.utils.share_memory_handler import redis_pool, \
    TO_PUBLISH_SEND_TASK, KILL_BEFORE_EXIT
# from hyades.utils.cpu_affinity import CPUAffinity

r = redis.Redis(connection_pool=redis_pool)


# class Client(ShareBase, CPUAffinity):
class Client(ShareBase):
    def __init__(self) -> None:
        self.client_id = Config().args.id
        ShareBase.__init__(self, client_id=self.client_id)
        # CPUAffinity.__init__(self)
        # if self.cpu_affinity_dict is not None:
        #     self.set_cpu_affinity(self.cpu_affinity_dict["comp"])

        # if we were in a simulation. then the server and clients share the same database
        # so only need to flush once
        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == "simple"):
            self.flush_db()
        else:  # but need to wait a while until the server flush
            time.sleep(3)  # TODO: avoid hard-coding

        self.log_prefix_str = f"[Client #{self.client_id} {os.getpid()}]"

        # related to secure aggregation
        self.sub, _ = self.subscribe_a_channel(
            channel_prefix=CLOSE_CLIENT
        )
        self.client_sampler = client_sampler_registry.get(
            client_id=self.client_id
        )
        sampling_rate_upperbound = self.client_sampler.get_sampling_rate_upperbound()
        num_sampled_clients_upperbound \
            = self.client_sampler.get_num_sampled_clients_upperbound()
        self.protocol = protocol_registry.get(
            client_id=self.client_id
        )
        # for setting DP parameters
        self.protocol.set_client_sampling_rate_upperbound(sampling_rate_upperbound)
        self.protocol.set_num_sampled_clients_upperbound(num_sampled_clients_upperbound)
        self.app = app_registry.get(client_id=self.client_id)

        data_dim = self.app.get_data_dim()
        self.data_dim = data_dim
        chunk_size = self.protocol.calc_chunk_size(data_dim)
        self.app.set_chunk_size(chunk_size=chunk_size)

        # if Config().app.type == "federated_learning":
        #     training_dataset_size = self.app.datasource.num_train_examples()
        #     self.set_a_shared_value(
        #         key=[TRAINING_DATASET_SIZE],
        #         value=training_dataset_size
        #     )

    def sending_and_cleaning(self):
        sub, ch_dict = self.batch_subscribe_channels(d={
            KILL_BEFORE_EXIT: False,
            TO_PUBLISH_SEND_TASK: True
        })
        kill_ch = ch_dict[KILL_BEFORE_EXIT]
        send_ch = ch_dict[TO_PUBLISH_SEND_TASK]

        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue

            channel = message["channel"].decode()
            channel = self.strip_self_channel_prefix(channel)
            if channel == send_ch:
                _data = pickle.loads(raw_data)
                key = _data['key']
                send_dict = self.get_a_shared_value(key=key)

                payload = send_dict["payload"]
                log_prefix_str = send_dict["log_prefix_str"]
                prompt = send_dict["prompt"]
                key_postfix = send_dict["key_postfix"]

                self.send(
                    payload=payload,
                    log_prefix_str=log_prefix_str,
                    key_postfix=key_postfix
                )
                if prompt:
                    log_content = f"{log_prefix_str} {prompt}".strip()
                    logging.info(log_content)
                self.delete_a_shared_value(key=key)
            elif channel == kill_ch:
                status_dict = self.protocol.get_status_dict()
                for key, status in status_dict.items():
                    pid = status["pid"]
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # the program has already dead
                    self.protocol.delete_a_status(key=key)
                break

    def send(self, payload, log_prefix_str, key_postfix):
        self._publish_a_value(
            channel=[SEND] + key_postfix,
            message={
                "payload": payload,
                "log_prefix_str": log_prefix_str
            },
            mode="large",
            subscriber_only_knows_prefix=True
        )

    def orchestrating(self):
        # sub.listen is blocking (the thread)
        for message in self.sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue

            self._publish_a_value(
                channel=KILL_BEFORE_EXIT,
                message=1  # signal
            )

            break

    def start_client(self) -> None:
        proc = ClientProcess(data_dim=self.data_dim)
        proc.start()

        t = threading.Thread(target=self.orchestrating)
        t.start()

        t = threading.Thread(target=self.sending_and_cleaning)
        t.start()

        self.app.run()
