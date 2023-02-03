import logging
from collections import OrderedDict
from hyades.primitives.authenticated_encryption import (
    fernet
)
from hyades.config import Config

registered_dict = OrderedDict([
    ('fernet', fernet.Handler)
])


def get():
    handler_type = Config().agg.security\
        .authenticated_encryption.type

    if handler_type in registered_dict:
        logging.info("Authenticated encryption: %s", handler_type)
        registered_handler = registered_dict[handler_type]()
    else:
        raise ValueError(
            f"No such authenticated encryption handler: {handler_type}")

    return registered_handler
