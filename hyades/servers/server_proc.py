import copy
import os
import re
import sys
import time
import redis
import pickle
import logging
import asyncio
import socketio
import aiohttp
import threading
from aiohttp import web
from distutils.version import StrictVersion
from multiprocessing import Process
from hyades.config import Config
from hyades.client_samplers import registry \
    as client_sampler_registry
from hyades.protocols import registry as protocol_registry
from hyades.utils.share_memory_handler import redis_pool, \
    CLOSE_SERVER, REMOVE_CLIENT, REGISTER_CLIENT, \
    PORT_SEND, ShareBase, SERVER_TO_CLIENT, CLIENT_TO_SERVER
from hyades.client_managers import registry \
    as client_manager_registry
# from hyades.utils.cpu_affinity import CPUAffinity

r = redis.Redis(connection_pool=redis_pool)


class ServerEvents(socketio.AsyncNamespace):
    def __init__(self, namespace, server_server):
        super().__init__(namespace)
        self.server_server = server_server
        self.log_prefix_string = f"[Server #{os.getpid()}]"

    async def on_connect(self, sid, environ):
        logging.info("%s A new client just connected to port %d.",
                     self.log_prefix_string, self.server_server.port)

    async def on_disconnect(self, sid):
        logging.info("%s An existing client just disconnected.",
                     self.log_prefix_string)
        await self.server_server.client_disconnected(sid)

    def on_client_alive(self, sid, data):
        self.server_server.register_client(sid, data)

    async def on_chunk(self, sid, data):
        await self.server_server.client_chunk_arrived(
            sid, data['data'], data['ref'])

    async def on_client_payload(self, sid, data):
        await self.server_server.client_payload_arrived(
            sid, data['id'], data['ref'])

    async def on_client_payload_done(self, sid, data):
        await self.server_server.client_payload_done(
            sid, data['id'], data['ref'])


# class ServerProcess(Process, ShareBase, CPUAffinity):
class ServerProcess(Process, ShareBase):
    def __init__(self, port, aux):
        ShareBase.__init__(self, client_id=0)
        Process.__init__(self)
        # CPUAffinity.__init__(self)
        # if self.cpu_affinity_dict is not None:
        #     self.set_cpu_affinity(self.cpu_affinity_dict["comm"])

        self.port = port
        self.clients = {}
        self.client_chunks = {}
        self.send_ref = {}
        self.client_payload = {}
        self.total_clients = Config().clients.total_clients
        self.log_prefix_str = None
        for client_id in range(1, self.total_clients + 1):
            self.send_ref[client_id] = 0

        self.client_sampler = client_sampler_registry.get()
        sampling_rate_upperbound = self.client_sampler.get_sampling_rate_upperbound()
        num_sampled_clients_upperbound \
            = self.client_sampler.get_num_sampled_clients_upperbound()
        self.protocol = protocol_registry.get()
        self.protocol.set_client_sampler(self.client_sampler)
        # for setting DP parameters
        self.protocol.set_client_sampling_rate_upperbound(sampling_rate_upperbound)
        self.protocol.set_num_sampled_clients_upperbound(num_sampled_clients_upperbound)

        # Because protocol needs to know "available_clients" when
        # sampling clients for next round
        self.client_manager = client_manager_registry.get()
        if Config().app.type == "federated_learning":  # the sampler needs sampling
            client_dataset_size_dict, = aux  # do not lose the comma
            self.client_manager \
                .set_client_dataset_size(client_dataset_size_dict)

        self.protocol.set_client_manager(self.client_manager)

    async def broadcast(self, payload, log_prefix_str, client_pair_list=None):
        assert client_pair_list is not None

        for physical_id, logical_id in client_pair_list:
            sid = self.clients[physical_id]['sid']
            customized_payload = copy.deepcopy(payload)
            customized_payload["logical_client_id"] = logical_id
            await self.send(
                sid=sid,
                payload=customized_payload,
                client_id=physical_id,
                log_prefix_str=log_prefix_str
            )

    def io_executing(self, loop):
        event_loop_for_sending = loop

        port_send_prefix = PORT_SEND + str(self.port)
        sub, ch_dict = self.batch_subscribe_channels(d={
            port_send_prefix: True,
            CLOSE_SERVER: False
        })
        port_send_ch = ch_dict[port_send_prefix]

        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue

            channel = message["channel"].decode()
            channel = self.strip_self_channel_prefix(channel)
            if channel == port_send_ch:
                _data = pickle.loads(raw_data)
                key = _data['key']
                # logging.info(f"[Debug] io_executing: {key}.")
                data = self.get_a_shared_value(key=key)
                self.delete_a_shared_value(key=key)
                payload = data["payload"]
                client_pair_list = data["client_pair_list"]
                log_prefix_str = data["log_prefix_str"]
                # logging.info(f"[Debug] Send and delete: {key}, {client_pair_list}.")
                # 1. Should not directly await here,
                # otherwise listen will be blocked
                # 2. Should not use create_task here
                # as the code is outside the event loop
                asyncio.run_coroutine_threadsafe(
                    coro=self.broadcast(
                        payload=payload,
                        log_prefix_str=log_prefix_str,
                        client_pair_list=client_pair_list
                    ),
                    loop=event_loop_for_sending
                )
            elif channel == CLOSE_SERVER:
                break

        # Should not use create_task here
        # as the code is outside the event loop
        asyncio.run_coroutine_threadsafe(
            coro=self.close(),
            loop=event_loop_for_sending
        )

    def run(self):
        self.log_prefix_str = f"[Server #{self.pid}]"

        ping_interval = Config().server.ping_interval if hasattr(
            Config().server, 'ping_interval') else 3600
        ping_timeout = Config().server.ping_timeout if hasattr(
            Config().server, 'ping_timeout') else 360
        self.sio = socketio.AsyncServer(
            # async_handlers=False,
            ping_interval=ping_interval,
            max_http_buffer_size=2 ** 31,
            ping_timeout=ping_timeout
        )
        self.sio.register_namespace(
            ServerEvents(namespace='/', server_server=self))

        loop = asyncio.get_event_loop()  # no get_running_loop
        # it must be in a new thread (event loop) as sub.listen is blocking
        t = threading.Thread(target=self.io_executing, args=(loop,))
        t.start()

        app = web.Application()
        self.sio.attach(app)
        if StrictVersion(aiohttp.__version__) < StrictVersion('3.8.0'):
            web.run_app(app, host=Config().server.address, port=self.port)
        else:
            web.run_app(app, host=Config().server.address, port=self.port,
                        loop=loop)

    def register_client(self, sid, data):
        client_id = data["id"]
        meta = data["meta"]
        if not client_id in self.clients:
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            logging.info("%s New client with id #%d arrived.", self.log_prefix_str,
                         client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            logging.info("%s New contact from Client #%d received.", self.log_prefix_str,
                         client_id)

        self._publish_a_value(
            channel=REGISTER_CLIENT,
            message={
                "client_id": client_id,
                "proc_port": self.port,
                "meta": meta
            }
        )

    async def close_connections(self):
        for client_id, client in dict(self.clients).items():
            logging.info("%s Closing the connection to client #%d.", self.log_prefix_str,
                         client_id)
            await self.sio.emit('disconnect', room=client['sid'])
            # await self.sio.call(
            #     event='disconnect',
            #     room=client['sid'],
            #     timeout=180
            # )

    async def send_in_chunks(self, data, sid, client_id, ref) -> None:
        step = 1024 ^ 2
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit('chunk', {
                'data': chunk,
                'ref': ref
            }, room=sid)
            # await self.sio.call(
            #     event='chunk',
            #     data={
            #         'data': chunk,
            #         'ref': ref
            #     },
            #     room=sid,
            #     timeout=180
            # )

        await self.sio.emit('payload', {
            'id': client_id,
            'ref': ref
        }, room=sid)
        # await self.sio.call(
        #     event='payload',
        #     data={
        #         'id': client_id,
        #         'ref': ref
        #     },
        #     room=sid,
        #     timeout=180
        # )

    def substitute_pid(self, log_prefix_str):
        return re.sub('\[Server #\d+\]',
                      f"[Server #{self.pid}]",
                      log_prefix_str)

    async def send(self, sid, payload, client_id, log_prefix_str) -> None:
        data_size = 0
        ref = self.send_ref[client_id]
        self.send_ref[client_id] += 1

        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == "simple"):
            if isinstance(payload, list):
                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data, sid,
                                              client_id, ref)
                    data_size += sys.getsizeof(_data)
            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data, sid,
                                          client_id, ref)
                data_size = sys.getsizeof(_data)
        else:  # simulation
            _data = pickle.dumps(payload)
            data_size = sys.getsizeof(_data)
            self.set_a_shared_value(
                key=[client_id, ref],
                value=payload,
                # need to be visible to clients
                customized_prefix=SERVER_TO_CLIENT
            )

        await self.sio.emit('payload_done', {
            'id': client_id,
            'ref': ref
        },
                            room=sid)
        # await self.sio.call(
        #     event='payload_done',
        #     data={
        #         'id': client_id,
        #         'ref': ref
        #     },
        #     room=sid,
        #     timeout=180
        # )
        log_prefix_str = self.substitute_pid(log_prefix_str)
        logging.info("%s Sent %s MB of payload data to physical client #%d.",
                     log_prefix_str, round(data_size / 1024**2, 6),
                     client_id)

    async def client_chunk_arrived(self, sid, data, ref) -> None:
        if sid not in self.client_chunks:
            self.client_chunks[sid] = {}
        if ref not in self.client_chunks[sid]:
            self.client_chunks[sid][ref] = []
        self.client_chunks[sid][ref].append(data)

    async def client_payload_arrived(self, sid, client_id, ref):
        assert len(self.client_chunks[sid][ref]) > 0

        payload = b''.join(self.client_chunks[sid][ref])
        _data = pickle.loads(payload)
        self.client_chunks[sid][ref] = []

        if sid not in self.client_payload \
                or self.client_payload[sid] is None:
            self.client_payload[sid] = {
                ref: _data
            }
        elif ref not in self.client_payload[sid] \
                or self.client_payload[sid][ref] is None:
            self.client_payload[sid][ref] = _data
        elif isinstance(self.client_payload[sid][ref], list):
            self.client_payload[sid][ref].append(_data)
        else:
            self.client_payload[sid][ref] = [
                self.client_payload[sid][ref]
            ]
            self.client_payload[sid][ref].append(_data)

    async def client_payload_done(self, sid, client_id, ref):
        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == "simple"):

            client_payload = self.client_payload[sid][ref]
            # accommodate high concurrency

            assert client_payload is not None
            assert client_id == client_payload["client_id"]
            # so that none can impersonate another

            payload_size = 0
            if isinstance(client_payload, list):
                for _data in client_payload:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(client_payload))

            self.protocol.handle_client_payload(
                client_id=client_id,
                payload=client_payload,
                payload_size=payload_size
            )
            del self.client_payload[sid][ref]
        else:
            client_payload = self.get_a_shared_value(
                key=[client_id, ref],
                # otherwise it is fetching its own variables
                customized_prefix=CLIENT_TO_SERVER
            )
            assert client_payload is not None
            assert client_id == client_payload["client_id"]

            payload_size = sys.getsizeof(pickle.dumps(client_payload))
            self.protocol.handle_client_payload(
                client_id=client_id,
                payload=client_payload,
                payload_size=payload_size
            )
            self.delete_a_shared_value(
                key=[client_id, ref],
                # otherwise it is deleting its own variables
                customized_prefix=CLIENT_TO_SERVER
            )

    async def client_disconnected(self, sid):
        for client_id, client in dict(self.clients).items():
            if client['sid'] == sid:
                del self.clients[client_id]

                logging.info(
                    "%s Client #%d disconnected and removed from this server.",
                    self.log_prefix_str, client_id)

                self._publish_a_value(
                    channel=REMOVE_CLIENT,
                    message={ "client_id": client_id }
                )

    async def close(self):
        logging.info("%s Closing the server.", self.log_prefix_str)
        await self.close_connections()
        os._exit(0)
