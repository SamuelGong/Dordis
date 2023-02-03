import logging
import socket
import time

import redis
import pickle
from hyades.config import Config

# variables shared among the server's processes
CLIENT_ATTENDANCE = "client_attendance"
SIMULATED_CLIENT_LATENCY = "simulated_client_latency"
CLIENT_STATS = "client_stats"
SAMPLED_CLIENTS = "sampled_clients"
TRACE_RELATED = "trace_related"

# variables shared among a client's processes
DELAY_MOCKING_FACTORS = "delay_mocking_factors"
NUM_SAMPLED_CLIENTS = "num_sampled_clients"
# TRAINING_DATASET_SIZE = "training_dataset_size"

# channels for server's subscribing
AGG_RES_PREPARED_FOR_SERVER = "agg_res_prepared_for_server"  # app
BROADCAST = "broadcast"  # io
CLOSE_SERVER = "close_server"  # misc
SCHEDULE = "schedule"  # scheduler
BREAK_SCHEDULER_LOOP = "break_scheduler_loop"  # scheduler

# channels for client's subscribing
AGG_RES_PREPARED_FOR_CLIENT = "agg_res_prepared_for_client"  # app
TO_PREPARE_DATA = "to_prepare_data"  # app
SEND = "send"  # io
CLOSE_CLIENT = "close_client"  # misc

# channels for client's publishing
DATA_PREPARED = "data_prepared"  # app
AGG_RES_USED_BY_CLIENT = 'agg_res_used_by_client'  # app

# channels for server's publishing
AGG_RES_USED_BY_SERVER = 'agg_res_used_by_server'  # app
REGISTER_CLIENT = "register_client"  # io
REMOVE_CLIENT = "remove_client"  # io
PORT_SEND = "port_send_"  # io

# channels for both clients and the server
KILL_BEFORE_EXIT = "kill_before_exit"  # send_and_clean
TO_PUBLISH_SEND_TASK = "to_publish_send_task"  # send_and_clean

# channels for simulation
SERVER_TO_CLIENT = 'server-to-client'
CLIENT_TO_SERVER = 'client_to_server'

redis_pool = redis.ConnectionPool(
    host='127.0.0.1',
    port=Config().server.redis.port,
    db=0,
    # health_check_interval=0.2,
    socket_keepalive=True,
    socket_keepalive_options={
        # socket.TCP_KEEPIDLE: 3600,
        socket.TCP_KEEPCNT: 2,
        socket.TCP_KEEPINTVL: 30
    }
)
r = redis.Redis(connection_pool=redis_pool)
# r = redis.Redis(
#     host='127.0.0.1',
#     port=Config().server.redis.port,
#     db=0,
#     socket_keepalive=True,
#     socket_keepalive_options={
#         socket.TCP_KEEPIDLE: 120,
#         socket.TCP_KEEPCNT: 2,
#         socket.TCP_KEEPINTVL: 30
#     }
# )


class ShareBase:
    def __init__(self, client_id):
        self.client_id = 0
        self._set_client_id(client_id)

    def _set_client_id(self, client_id):
        self.client_id = client_id
        if client_id == 0:
            self.r_prefix = "server/"
        else:
            self.r_prefix = f"client-{self.client_id}/"

    def _form_redis_key(self, l, prefix=False):
        l = [str(e) for e in l]
        body = "/".join(l)
        if not prefix:
            return body
        else:
            return body + '/*'

    def _keys_of_a_prefix(self, prefix):
        if isinstance(prefix, list):
            prefix = self._form_redis_key(
                l=prefix,
                prefix=True
            )

        # Note that scanning can be too slow!
        # So do not abuse this function.
        l = [e.decode() for e in r.scan_iter(self.r_prefix + prefix)]
        for i, item in enumerate(l):
            l[i] = self.strip_self_channel_prefix(item)
        return l

    def _set_a_value(self, key, value, customized_prefix=None):
        if isinstance(key, list):
            key = self._form_redis_key(l=key)

        if customized_prefix:
            r.set(
                name=customized_prefix + key,
                value=pickle.dumps(value)
            )
        else:
            r.set(
                name=self.r_prefix + key,
                value=pickle.dumps(value)
            )

    def _get_a_value(self, key, busy_waiting=False,
                     customized_prefix=None):
        if isinstance(key, list):
            key = self._form_redis_key(l=key)
        if customized_prefix:
            prefix = customized_prefix
        else:
            prefix = self.r_prefix

        raw_value = r.get(prefix + key)
        if raw_value is None:
            if busy_waiting:
                while not raw_value:
                    time.sleep(0.5)
                    raw_value = r.get(prefix + key)
                return pickle.loads(data=raw_value)
            else:
                return None
        else:
            return pickle.loads(data=raw_value)

    def _delete_a_key(self, key, customized_prefix=None):
        if isinstance(key, list):
            key = self._form_redis_key(l=key)

        if customized_prefix:
            r.delete(customized_prefix + key)
        else:
            r.delete(self.r_prefix + key)

    def _parse_subscribe_channel(self, channel_prefix,
                                 is_large_value=False):
        if is_large_value:
            ch = self.get_channel_for_subscribing_a_large_value(
                channel_prefix=channel_prefix
            )
        else:
            if isinstance(channel_prefix, list):
                ch = self._form_redis_key(channel_prefix)
            else:
                ch = channel_prefix

        return ch

    def strip_self_channel_prefix(self, channel):
        i = channel.find(self.r_prefix) + len(self.r_prefix)
        return channel[i:]

    def subscribe_a_channel(self, channel_prefix,
                            is_large_value=False):
        sub = r.pubsub()
        ch = self._parse_subscribe_channel(
            channel_prefix=channel_prefix,
            is_large_value=is_large_value
        )
        sub.subscribe(self.r_prefix + ch)
        return sub, ch

    def batch_subscribe_channels(self, d):
        sub = r.pubsub()

        ch_dict = {}
        ch_list = []
        for channel_prefix, is_large_value in d.items():
            ch = self._parse_subscribe_channel(
                channel_prefix=channel_prefix,
                is_large_value=is_large_value
            )
            ch_list.append(self.r_prefix + ch)
            ch_dict[channel_prefix] = ch

        sub.subscribe(*ch_list)
        return sub, ch_dict

    def set_a_shared_value(self, key, value,
                           customized_prefix=None):
        self._set_a_value(
            key=key,
            value=value,
            customized_prefix=customized_prefix
        )

    def batch_set_shared_values(self, d, postfix):
        for key, value in d.items():
            key = [key] + postfix
            self.set_a_shared_value(
                key=key,
                value=value
            )

    def get_a_shared_value(self, key, busy_waiting=False,
                           customized_prefix=None):
        return self._get_a_value(key=key, busy_waiting=busy_waiting,
                                 customized_prefix=customized_prefix)

    def batch_get_shared_values(self, keys, postfix):
        result = []
        for key in keys:
            result.append(self.get_a_shared_value(
                key=[key] + postfix
            ))
        return result

    def delete_a_shared_value(self, key, customized_prefix=None):
        self._delete_a_key(key=key,
                           customized_prefix=customized_prefix)

    def batch_delete_shared_values(self, keys, postfix):
        for key in keys:
            key = [key] + postfix
            self.delete_a_shared_value(key=key)

    def delete_a_prefix(self, prefix):
        for key in self._keys_of_a_prefix(prefix=prefix):
            self._delete_a_key(key=key)

    def prefix_to_dict(self, prefix, key_type="int"):
        keys = self._keys_of_a_prefix(prefix=prefix)
        if key_type == 'int':
            return {
                int(e.split('/')[-1]):
                    self._get_a_value(key=e)
                for e in keys
            }
        else:  # 'str'
            return {
                e.split('/')[-1]: self._get_a_value(key=e)
                for e in keys
            }

    def _publish_a_value(self, channel, message, mode="small",
                         subscriber_only_knows_prefix=False):
        if mode == "small":
            if isinstance(channel, list):
                channel = self._form_redis_key(channel)

            r.publish(
                channel=self.r_prefix + channel,
                message=pickle.dumps(message)
            )
        else:  # mode == "large"
            if isinstance(channel, list):
                channel_prefix = channel[0]
                channel = self._form_redis_key(channel)
            else:
                channel_prefix = channel

            self._set_a_value(
                key=channel + '/body',
                value=message
            )

            if subscriber_only_knows_prefix:
                r.publish(
                    channel=self.r_prefix
                            + channel_prefix + '/signal',
                    message=pickle.dumps({
                        'key': channel + '/body'
                    })
                )
            else:
                r.publish(
                    channel=self.r_prefix
                            + channel + '/signal',
                    message=1  # place holder
                )

    def get_channel_for_subscribing_a_large_value(
            self, channel_prefix):
        if isinstance(channel_prefix, list):
            channel_prefix = self._form_redis_key(channel_prefix)

        return channel_prefix + '/signal'

    def get_key_for_fetching_a_large_value(self, key_prefix):
        if isinstance(key_prefix, list):
            key_prefix = self._form_redis_key(key_prefix)

        return key_prefix + '/body'

    def flush_db(self):
        r.flushdb()
