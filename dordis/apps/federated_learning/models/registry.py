import logging
from collections import OrderedDict
from dordis.config import Config
from dordis.apps.federated_learning.models import (
    lenet5, vgg, resnet, cnn, shufflenet, efficientnet
)

registered_models = OrderedDict([
    ('lenet5', lenet5.Model),
    ('vgg', vgg.Model),
    ('resnet', resnet.Model),
    ('cnn', cnn.Model),
    ('shufflenet', shufflenet.Model),
    ('efficientnet', efficientnet.Model)
])


def get(num_classes=10):
    model_name = Config().app.trainer.model_name
    model_type = model_name.split('_')[0]
    model = None

    if hasattr(Config().app.trainer, "num_classes"):
        num_classes = Config().app.trainer.num_classes

    for name, registered_model in registered_models.items():
        if name.startswith(model_type):
            model = registered_model.get_model(model_name, num_classes)
            logging.info(f"Got {model_name} model "
                         f"with {num_classes}-dim output.")

    if model is None:
        raise ValueError('No such FL model: {}'.format(model_name))

    return model
