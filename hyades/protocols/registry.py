import logging
from collections import OrderedDict

from hyades.protocols import (
    plaintext,
    secagg,
    dp,
    dp_plus_secagg
)

from hyades.config import Config

registered_server = OrderedDict([
    ('plaintext', plaintext.ProtocolServer),
    ('secagg', secagg.ProtocolServer),
    ('dp', dp.ProtocolServer),
    ('dp_plus_secagg', dp_plus_secagg.ProtocolServer),
])

registered_clients = OrderedDict([
    ('plaintext', plaintext.ProtocolClient),
    ('secagg', secagg.ProtocolClient),
    ('dp', dp.ProtocolClient),
    ('dp_plus_secagg', dp_plus_secagg.ProtocolClient),
])


def get(client_id=None):
    protocol_type = Config().agg.type

    if client_id is None:
        name = "Server protocol"
        registered_dict = registered_server
        client_id = 0
    else:
        name = "Client protocol"
        registered_dict = registered_clients

    if protocol_type in registered_dict:
        logging.info("%s: %s", name, protocol_type)
        registered_handler = registered_dict[protocol_type](client_id)
    else:
        raise ValueError(f"{name}: No such aggregation protocol: "
                         f"{protocol_type}")

    return registered_handler
