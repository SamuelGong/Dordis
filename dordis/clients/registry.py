import logging
from collections import OrderedDict

from dordis.clients import (
    base,
)

from dordis.config import Config


registered_clients = OrderedDict([
    ('base', base.Client),
])


def get():
    client_type = Config().clients.type

    if client_type in registered_clients:
        logging.info("Client: %s", client_type)
        registered_client = registered_clients[client_type]()
    else:
        raise ValueError('No such client: {}'.format(client_type))

    return registered_client
