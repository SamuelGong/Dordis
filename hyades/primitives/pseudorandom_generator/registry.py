import logging
from collections import OrderedDict
from hyades.primitives.pseudorandom_generator import (
    os_random
)
from hyades.config import Config

registered_dict = OrderedDict([
    ('os_random', os_random.Handler)
])


def get():
    handler_type = Config().agg.security\
        .pseudorandom_generator.type

    if handler_type in registered_dict:
        logging.info("Pseudorandom generator: %s", handler_type)
        registered_handler = registered_dict[handler_type]()
    else:
        raise ValueError(
            f"No such pseudorandom generator handler: {handler_type}")

    return registered_handler
