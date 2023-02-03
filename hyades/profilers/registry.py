import logging
from collections import OrderedDict
from hyades.profilers import (
    base
)
from hyades.config import Config

registered_dict = OrderedDict([
    ('base', base.Profiler)
])


def get():
    handler_type = Config().scheduler.profiler.type

    if handler_type in registered_dict:
        logging.info("Aggregation profiler: %s", handler_type)
        registered_profiler = registered_dict[handler_type]()
    else:
        raise ValueError(
            f"No such aggregation profiler: {handler_type}")

    return registered_profiler
