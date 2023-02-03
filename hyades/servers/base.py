import logging
import os
import gc
import time
import redis
import queue
import pickle
import signal
import threading
import multiprocessing as mp
from hyades.apps import registry as app_registry
from hyades.schedulers import registry as scheduler_registry
from hyades.protocols import registry as protocol_registry
from hyades.client_samplers import registry as client_sampler_registry
from hyades.client_samplers.base import ClientSamplePlugin
from hyades.client import run
from hyades.config import Config
# from hyades.utils import csv_processor
# from hyades.utils.misc import get_chunks_idx
from hyades.servers.server_proc import ServerProcess
from hyades.utils.share_memory_handler import redis_pool, \
    REGISTER_CLIENT, PORT_SEND, REMOVE_CLIENT, TO_PUBLISH_SEND_TASK, \
    CLOSE_SERVER, KILL_BEFORE_EXIT
from hyades.client_managers import registry \
    as client_manager_registry
# from hyades.utils.cpu_affinity import CPUAffinity

r = redis.Redis(connection_pool=redis_pool)


# class Server(ShareBase, CPUAffinity):
# no need to inherit ShareBase as ClientSamplePlugin already does so
class Server(ClientSamplePlugin):
    def __init__(self):
        ClientSamplePlugin.__init__(self, client_id=0)
        # CPUAffinity.__init__(self)
        # if self.cpu_affinity_dict is not None:
        #     self.set_cpu_affinity(self.cpu_affinity_dict["comp"])
        self.flush_db()

        self.simulation = hasattr(Config(), "simulation")
        logging.info(f"Simulation: {self.simulation}")

        self.clients = []
        self.total_clients = Config().clients.total_clients

        self.client_proc_port_mapping = {}
        self.log_prefix_str = f"[Server #{os.getpid()}]"
        self.sub, _ = self.batch_subscribe_channels(d={
            REGISTER_CLIENT: False,
            REMOVE_CLIENT: False,
            CLOSE_SERVER: False
        })

        # when in resource saving mode, we only
        # use self.num_physical_clients physical clients to
        # emulate self.total_clients clients
        if hasattr(Config().clients, "resource_saving") \
                and Config().clients.resource_saving:
            assert hasattr(Config().clients, "num_physical_clients")
            self.num_physical_clients = Config().clients.num_physical_clients
            self.init_scale = int(Config().app.init_scale_threshold
                                  * self.num_physical_clients)
        else:
            self.num_physical_clients = self.total_clients
            self.init_scale = int(Config().app.init_scale_threshold
                                  * self.total_clients)

        recorded_items = Config().results.types
        self.recorded_items = [
            x.strip() for x in recorded_items.split(',')
        ]
        self.initialized = False
        self.no_more_spawn = False

        self.client_sampler = client_sampler_registry.get(
            client_id=self.client_id
        )
        sampling_rate_upperbound = self.client_sampler.get_sampling_rate_upperbound()
        num_sampled_clients_upperbound \
            = self.client_sampler.get_num_sampled_clients_upperbound()

        self.protocol = protocol_registry.get()
        self.protocol.set_client_sampler(self.client_sampler)
        # for setting DP parameters
        self.protocol.set_client_sampling_rate_upperbound(sampling_rate_upperbound)
        self.protocol.set_num_sampled_clients_upperbound(num_sampled_clients_upperbound)
        graph_dict = self.protocol.get_graph_dict()

        self.schedule = queue.Queue()
        self.scheduler = scheduler_registry.get(
            hyades_instance=self,
            log_prefix_str=self.log_prefix_str,
        )
        self.scheduler.register_graph(graph_dict)

        self.app = app_registry.get()
        data_dim = self.app.get_data_dim()
        chunk_size = self.protocol.calc_chunk_size(data_dim)  # must be after r.flushdb
        self.app.set_chunk_size(chunk_size)
        self.num_chunks = len(chunk_size)
        self.scheduler.set_num_chunks(num_chunks=self.num_chunks)
        self.client_manager = client_manager_registry.get()

        if Config().app.type == "federated_learning":
            client_dataset_size_dict \
                = self.app.get_client_dataset_size_dict(
                clients=range(1, self.total_clients + 1),
                log_prefix_str=self.log_prefix_str
            )
            self.client_dataset_size_dict = client_dataset_size_dict
            self.client_manager\
                .set_client_dataset_size(client_dataset_size_dict)
        self.protocol.set_client_manager(self.client_manager)

    def register_schedule(self, schedule):
        for item in schedule:
            self.schedule.put(item)

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
                send_list = send_dict["list"]
                send_list_round_idx = send_dict["list_round_idx"]

                # round_idx is needed for supporting client subsampling
                # as well as mocking dropout
                if send_list_round_idx is not None:
                    # in applications like FL, we sometimes need to
                    # send to clients who are sampled in other round
                    round_idx = send_list_round_idx
                else:
                    # round_idx happens to be included in key_postfix
                    # so we leverage this fact
                    round_idx = key_postfix[0]

                type = send_dict["type"]
                if type == "default":
                    self.broadcast_payload(
                        payload=payload,
                        round_idx=round_idx,
                        log_prefix_str=log_prefix_str,
                        send_list=send_list,
                        key_postfix=key_postfix
                    )
                elif type == "exchange":
                    self.exchange_payload(
                        exchange_payload_dict=payload,
                        round_idx=round_idx,
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

    def start_scheduler(self):
        self.scheduler.schedule()

        message = 'Closing the server.'
        self.close(message=message)

    def close(self, message):
        logging.info("%s %s", self.log_prefix_str, message)
        self._publish_a_value(
            channel=CLOSE_SERVER,
            message={'message': message }
        )
        while not self.no_more_spawn:
            continue
        self._publish_a_value(
            channel=KILL_BEFORE_EXIT,
            message=1  # signal
        )

    def orchestrating(self):
        # sub.listen is blocking (the thread)
        for message in self.sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue

            channel = message["channel"].decode()
            channel = self.strip_self_channel_prefix(channel)
            data = pickle.loads(raw_data)

            if channel == REGISTER_CLIENT:
                self.register_client(
                    client_id=data["client_id"],
                    proc_port=data["proc_port"],
                    meta=data["meta"]
                )
            elif channel == CLOSE_SERVER:
                break
            elif channel == REMOVE_CLIENT:
                self.remove_client(
                    client_id=data["client_id"]
                )

        self.no_more_spawn = True

    def run(self, port=Config().server.port):
        if not isinstance(port, list):
            port = [port]

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        for p in port:
            logging.info("%s Starting a server at %s:%s.",
                         self.log_prefix_str, Config().server.address, p)
            if Config().app.type == "federated_learning":
                aux = (self.client_dataset_size_dict,)
            else:
                aux = None
            proc = ServerProcess(port=p, aux=aux)
            proc.start()

        t = threading.Thread(target=self.orchestrating)
        t.start()

        t = threading.Thread(target=self.sending_and_cleaning)
        t.start()

        self.app.run()

        # prevent loss of money due to scp errors (config.yml)
        logging.info(f"Waiting for clients to participate.")
        wait_clients_time = 180  # TODO: avoid hard-coding
        time.sleep(wait_clients_time)
        if len(self.clients) < self.init_scale:
            message = f'Not enough clients ({len(self.clients)} ' \
                      f'< {self.init_scale}) participate timely.'
            self.close(message=message)

    def start_clients(self):
        starting_id = 1

        total_processes = self.num_physical_clients
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        for client_id in range(starting_id, total_processes + starting_id):
            logging.info("%s Starting client #%d's process.",
                         self.log_prefix_str, client_id)
            proc = mp.Process(target=run,
                              args=(client_id, None))
            proc.start()

    def _broadcast_payload(self, payload, round_idx, log_prefix_str,
                          key_postfix, send_list=None):
        # plugins for client sampling
        # [(physical_id, logical_id)]
        client_pair_list = [(self.client_id_transform(
            client_id=logical_client_id,
            round_idx=round_idx,
            mode="to_physical"
        ), logical_client_id) for logical_client_id in send_list]

        proc_port_client_mapping = {}
        for client_id_pair in client_pair_list:
            proc_port = self.client_proc_port_mapping[client_id_pair[0]]
            if proc_port not in proc_port_client_mapping:
                proc_port_client_mapping[proc_port] = []
            proc_port_client_mapping[proc_port].append(client_id_pair)

        for proc_port, client_pair_list in proc_port_client_mapping.items():
            self._publish_a_value(
                channel=[PORT_SEND + str(proc_port)] + key_postfix,
                message={
                    "payload": payload,
                    "client_pair_list": client_pair_list,
                    "log_prefix_str": log_prefix_str,
                },
                mode="large",
                subscriber_only_knows_prefix=True
            )
            # logging.info(f"[Debug] Publishing {[PORT_SEND + str(proc_port)] + key_postfix}.")

    def broadcast_payload(self, payload, round_idx, log_prefix_str,
                          key_postfix, send_list=None):
        assert send_list is not None
        # can happen, related to mocking dropout. see
        # server's handler_tail for more details
        if len(send_list) == 0:
            return

        # else:
        self._broadcast_payload(
            payload=payload,
            round_idx=round_idx,
            log_prefix_str=log_prefix_str,
            key_postfix=key_postfix,
            send_list=send_list
        )

    def exchange_payload(self, exchange_payload_dict, round_idx,
                         log_prefix_str, key_postfix):
        for client_id, data in exchange_payload_dict.items():
            self.broadcast_payload(
                payload=data,
                round_idx=round_idx,
                log_prefix_str=log_prefix_str,
                # prevent being deleted unexpectedly
                key_postfix=key_postfix + [client_id],
                send_list=[client_id],
            )

    def register_client(self, client_id, proc_port, meta):
        self.clients.append(client_id)
        self.client_proc_port_mapping[client_id] = proc_port
        logging.info(f"{self.log_prefix_str} Physical client {client_id} "
                     f"registers and is bond to port {proc_port}.")

        if len(self.clients) >= self.init_scale and not self.initialized:
            self.initialized = True
            logging.info("%s Starting working.", self.log_prefix_str)
            t = threading.Thread(target=self.start_scheduler)
            t.start()

    def remove_client(self, client_id):
        self.clients.remove(client_id)
        del self.client_proc_port_mapping[client_id]
