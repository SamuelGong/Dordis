import os
import re
import sys
import time
import redis
import pickle
import logging
import asyncio
import socketio
import threading
# from hyades.utils.cpu_affinity import CPUAffinity
from multiprocessing import Process
from hyades.config import Config
from hyades.protocols import registry as protocol_registry
from hyades.client_samplers import registry as client_sampler_registry
from hyades.utils.share_memory_handler import redis_pool, \
    ShareBase, SEND, CLOSE_CLIENT, SERVER_TO_CLIENT, \
    CLIENT_TO_SERVER

r = redis.Redis(connection_pool=redis_pool)


class ClientEvents(socketio.AsyncClientNamespace):
    def __init__(self, namespace, hyades_client):
        super().__init__(namespace)
        self.hyades_client = hyades_client
        self.client_id = hyades_client.client_id
        self.log_prefix_str = f"[Client #{self.client_id} {os.getpid()}]"

    def on_connect(self):
        logging.info("%s Connected to the server.", self.log_prefix_str)

    def on_disconnect(self):
        logging.info("%s The server disconnected the connection.",
                     self.log_prefix_str)
        self.hyades_client.close()

    def on_connect_error(self, data):
        logging.info("%s A connection attempt to the server failed.",
                     self.log_prefix_str)

    async def on_chunk(self, data):
        await self.hyades_client.chunk_arrived(data['data'], data['ref'])

    async def on_payload(self, data):
        await self.hyades_client.payload_arrived(data['id'], data['ref'])

    def on_payload_done(self, data):
        self.hyades_client.payload_done(data['id'], data['ref'])


# class ClientProcess(Process, ShareBase, CPUAffinity):
class ClientProcess(Process, ShareBase):
    def __init__(self, data_dim):
        self.client_id = Config().args.id
        ShareBase.__init__(self, self.client_id)
        Process.__init__(self)
        # CPUAffinity.__init__(self)
        # if self.cpu_affinity_dict is not None:
        #     self.set_cpu_affinity(self.cpu_affinity_dict["comm"])

        self.sio = None
        self.chunks = {}
        self.log_prefix_str = f"[Client #{self.client_id} {os.getpid()}]"
        self.server_payload = {}
        self.send_ref = 0

        self.client_sampler = client_sampler_registry.get(client_id=self.client_id)
        sampling_rate_upperbound = self.client_sampler.get_sampling_rate_upperbound()
        num_sampled_clients_upperbound \
            = self.client_sampler.get_num_sampled_clients_upperbound()
        self.protocol = protocol_registry.get(client_id=self.client_id)
        # for calculating DP parameters
        self.protocol.set_client_sampling_rate_upperbound(sampling_rate_upperbound)
        self.protocol.set_num_sampled_clients_upperbound(
            num_sampled_clients_upperbound)
        self.protocol.calc_chunk_size(data_dim)

    async def _run(self):
        logging.info("%s Contacting the central server.",
                     self.log_prefix_str)

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(
            ClientEvents(namespace='/', hyades_client=self))

        if hasattr(Config().server, 'use_https'):
            uri = 'https://{}'.format(Config().server.address)
        else:
            uri = 'http://{}'.format(Config().server.address)

        if hasattr(Config().server, 'port'):
            uri = '{}:{}'.format(uri, Config().server.port)

        logging.info("%s Connecting to the server at %s.",
                     self.log_prefix_str, uri)

        connection_gap = 1
        while True:
            try:
                await self.sio.connect(uri)
            except socketio.exceptions.ConnectionError as e:
                logging.info(
                    "%s The server is not online (%s). "
                    "Try to reconnect %ss later.", self.log_prefix_str,
                    e, connection_gap)
                time.sleep(connection_gap)
            except Exception as e:
                raise e
            else:
                break

        # await self.sio.emit(
        #     event='client_alive',
        #     data={'id': self.client_id},
        # )

        # getting metadata
        meta = {}
        await self.sio.call(
            event='client_alive',
            data={
                'id': self.client_id,
                'meta': meta
            },
            timeout=1800
        )
        await self.sio.wait()

    def run(self):
        loop = asyncio.get_event_loop()

        # it must be in a new thread (event loop) as sub.listen is blocking
        t = threading.Thread(target=self.io_executing, args=(loop,))
        t.start()

        loop.run_until_complete(self._run())

    def io_executing(self, loop):
        event_loop_for_sending = loop

        sub, ch_dict = self.batch_subscribe_channels(d={
            SEND: True,
            CLOSE_CLIENT: False
        })
        send_ch = ch_dict[SEND]
        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue

            channel = message["channel"].decode()
            channel = self.strip_self_channel_prefix(channel)

            if channel == send_ch:
                _data = pickle.loads(raw_data)
                key = _data['key']
                data = self.get_a_shared_value(key=key)
                self.delete_a_shared_value(key=key)
                payload = data["payload"]
                log_prefix_str = data["log_prefix_str"]

                # 1. Should not directly await here,
                # otherwise listen will be blocked
                # 2. Should not use create_task here
                # as the code is outside the event loop
                asyncio.run_coroutine_threadsafe(
                    coro=self.send(
                        payload=payload,
                        log_prefix_str=log_prefix_str
                    ),
                    loop=event_loop_for_sending
                )
            elif channel == CLOSE_CLIENT:
                break

    def close(self):
        message = "Instructed to exit by the server."
        logging.info(f"%s %s", self.log_prefix_str, message)

        self._publish_a_value(
            channel=CLOSE_CLIENT,
            message={'message': message}
        )
        os._exit(0)

    async def send_in_chunks(self, data, ref) -> None:
        step = 1024 ^ 2
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            # await self.sio.emit(
            #     event='chunk',
            #     data={'data': chunk, 'ref': ref},
            # )
            await self.sio.call(
                event='chunk',
                data={'data': chunk, 'ref': ref},
                timeout=1800
            )
        # await self.sio.emit(
        #     event='client_payload',
        #     data={'id': self.client_id, 'ref': ref},
        # )
        await self.sio.call(
            event='client_payload',
            data={'id': self.client_id, 'ref': ref},
            timeout=1800
        )

    def substitute_pid(self, log_prefix_str):
        return re.sub(f'\[Client #{self.client_id} #\d+\]',
                      f"[Client #{self.client_id} #{self.pid}]",
                      log_prefix_str)

    async def send(self, payload, log_prefix_str) -> None:
        ref = self.send_ref
        self.send_ref += 1

        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == "simple"):
            if isinstance(payload, list):
                data_size: int = 0

                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data, ref)
                    data_size += sys.getsizeof(_data)
            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data, ref)
                data_size = sys.getsizeof(_data)
        else:  # simulation
            _data = pickle.dumps(payload)
            data_size = sys.getsizeof(_data)
            self.set_a_shared_value(
                key=[self.client_id, ref],
                value=payload,
                # need to be visible to the server
                customized_prefix=CLIENT_TO_SERVER
            )

        # await self.sio.emit(
        #     event='client_payload_done',
        #     data = {
        #         'id': self.client_id,
        #         'ref': ref
        #     }
        # )
        await self.sio.call(
            event='client_payload_done',
            data={
                'id': self.client_id,
                'ref': ref
            },
            timeout=1800
        )
        log_prefix_str = self.substitute_pid(log_prefix_str)
        logging.info("%s Sent %s MB of payload data to the server.",
                     log_prefix_str, round(data_size / 1024**2, 6))

    async def chunk_arrived(self, data, ref) -> None:
        if ref not in self.chunks:
            self.chunks[ref] = []
        self.chunks[ref].append(data)

    async def payload_arrived(self, client_id, ref) -> None:
        assert client_id == self.client_id

        payload = b''.join(self.chunks[ref])
        _data = pickle.loads(payload)
        self.chunks[ref] = []

        if ref not in self.server_payload \
                or self.server_payload[ref] is None:
            self.server_payload[ref] = _data
        elif isinstance(self.server_payload[ref], list):
            self.server_payload[ref].append(_data)
        else:
            self.server_payload[ref] = [self.server_payload[ref]]
            self.server_payload[ref].append(_data)

    def payload_done(self, client_id, ref) -> None:
        assert client_id == self.client_id

        if not (hasattr(Config(), "simulation")
                and Config().simulation.type == "simple"):
            payload_size = 0

            if isinstance(self.server_payload[ref], list):
                for _data in self.server_payload[ref]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            elif isinstance(self.server_payload[ref], dict):
                for key, value in self.server_payload[ref].items():
                    payload_size += sys.getsizeof(pickle.dumps({key: value}))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.server_payload[ref]))

            self.protocol.handle_server_payload(
                payload=self.server_payload[ref],
                payload_size=payload_size
            )
            del self.server_payload[ref]
        else:  # simulation
            payload = self.get_a_shared_value(
                key=[client_id, ref],
                # otherwise it is fetching its own variables
                customized_prefix=SERVER_TO_CLIENT
            )
            payload_size = sys.getsizeof(pickle.dumps(payload))
            self.protocol.handle_server_payload(
                payload=payload,
                payload_size=payload_size
            )
            self.delete_a_shared_value(
                key=[client_id, ref],
                # otherwise it is deleting its own variables
                customized_prefix=SERVER_TO_CLIENT
            )
