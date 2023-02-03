import logging
from collections import OrderedDict
from hyades.schedulers import (
    base
)
from hyades.config import Config

registered_dict = OrderedDict([
    ('base', base.Scheduler)
])


def get(hyades_instance, log_prefix_str):
    handler_type = Config().scheduler.type

    if handler_type in registered_dict:
        logging.info("Aggregation scheduler: %s", handler_type)
        registered_scheduler = registered_dict[handler_type](
            hyades_instance=hyades_instance,
            log_prefix_str=log_prefix_str
        )
    else:
        raise ValueError(
            f"No such aggregation scheduler: {handler_type}")

    return registered_scheduler
