import logging
from collections import OrderedDict
from dordis.primitives.secret_sharing import (
    shamir, myshamir
)
from dordis.config import Config

registered_dict = OrderedDict([
    ('shamir', shamir.Handler),
    ('myshamir', myshamir.Handler)
])


def get():
    handler_type = Config().agg.security.secret_sharing.type

    if handler_type in registered_dict:
        # logging.info("Secret sharing: %s", handler_type)
        registered_handler = registered_dict[handler_type]()
    else:
        raise ValueError(
            f"No such secret sharing handler: {handler_type}")

    return registered_handler
