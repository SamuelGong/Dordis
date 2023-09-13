from collections import OrderedDict
from dordis.config import Config
from dordis.apps.federated_learning.trainers import (
    basic,
)


registered_trainers = OrderedDict([
    ('basic', basic.Trainer),
])


def get(model=None):
    trainer_name = Config().app.trainer.type

    if trainer_name in registered_trainers:
        registered_trainer = registered_trainers[trainer_name](model)
    else:
        raise ValueError('No such FL trainer: {}'.format(trainer_name))

    return registered_trainer
