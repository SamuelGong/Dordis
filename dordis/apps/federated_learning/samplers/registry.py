from collections import OrderedDict
from dordis.config import Config
from dordis.apps.federated_learning.samplers import (
    iid, all_inclusive, dirichlet
)

registered_samplers = OrderedDict([
    ('iid', iid.Sampler),
    ('all_inclusive', all_inclusive.Sampler),
    ('noniid', dirichlet.Sampler),
])


def get(datasource, client_id):
    if hasattr(Config().app.data, 'sampler'):
        sampler_type = Config().app.data.sampler
    else:
        sampler_type = 'iid'

    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](datasource,
                                                               client_id)
    else:
        raise ValueError('No such FL sampler: {}'.format(sampler_type))

    return registered_sampler
