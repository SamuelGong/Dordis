import logging
from collections import OrderedDict

from hyades.client_samplers import (
    all_inclusive, uniform, trace_driven, oort
)

from hyades.config import Config


registered_clients = OrderedDict([
    ('all_inclusive', all_inclusive.ClientSampler),
    ('uniform', uniform.ClientSampler),
    ("trace_driven", trace_driven.ClientSampler),
    ("oort", oort.ClientSampler)
])


def get(client_id=0):
    client_sampler_type = "all_inclusive"
    if hasattr(Config().clients, "sample"):
        client_sampler_type = Config().clients.sample.type

    if client_sampler_type in registered_clients:
        logging.info("Client sampler: %s", client_sampler_type)
        registered_client = registered_clients[client_sampler_type](client_id)
    else:
        raise ValueError('No such client samplers: {}'.format(client_sampler_type))

    return registered_client
