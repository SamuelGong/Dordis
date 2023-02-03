import logging
from collections import OrderedDict

from hyades.config import Config

from hyades.apps.federated_learning.datasources import (
    mnist, femnist, cinic10, cifar10, cifar100
)

registered_datasources = OrderedDict([
    ('MNIST', mnist),
    ('CINIC10', cinic10),
    ('CIFAR10', cifar10),
    ('CIFAR100', cifar100)
])

registered_partitioned_datasources = OrderedDict([
    ('FEMNIST', femnist),
])


def get(client_id=0, quiet=False):
    datasource_name = Config().app.data.datasource

    if not quiet:
        logging.info("Data source: %s", Config().app.data.datasource)

    if datasource_name in registered_datasources:
        dataset = registered_datasources[datasource_name].DataSource(client_id, quiet)
    elif datasource_name in registered_partitioned_datasources:
        dataset = registered_partitioned_datasources[
            datasource_name].DataSource(client_id, quiet)
    else:
        raise ValueError('No such FL data source: {}'
                         .format(datasource_name))

    return dataset
