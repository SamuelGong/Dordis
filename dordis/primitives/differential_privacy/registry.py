import logging
from collections import OrderedDict
from dordis.primitives.differential_privacy import (
    ddgauss, dskellam
)
from dordis.config import Config

registered_dict = OrderedDict([
    ('ddgauss', ddgauss.Handler),
    ('dskellam', dskellam.Handler)
])


def get():
    handler_type = Config().agg\
        .differential_privacy.mechanism

    if handler_type in registered_dict:
        logging.info("Differential privacy: %s", handler_type)
        registered_handler = registered_dict[handler_type]()
    else:
        raise ValueError(
            f"No such DP mechanism: {handler_type}")

    return registered_handler
