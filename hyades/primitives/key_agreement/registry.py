import logging
from collections import OrderedDict
from hyades.primitives.key_agreement import (
    elliptic_curve
)
from hyades.config import Config

registered_dict = OrderedDict([
    ('elliptic_curve', elliptic_curve.Handler)
])


def get():
    handler_type = Config().agg.security.key_agreement.type

    if handler_type in registered_dict:
        logging.info("Key agreement: %s", handler_type)
        registered_handler = registered_dict[handler_type]()
    else:
        raise ValueError(
            f"No such key agreement handler: {handler_type}")

    return registered_handler
