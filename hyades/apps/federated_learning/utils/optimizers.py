from torch import optim
from torch import nn

from hyades.config import Config


def get_optimizer(model) -> optim.Optimizer:
    weight_decay = Config().app.trainer.weight_decay
    model_name = Config().app.trainer.model_name
    if 'albert' in model_name:
        no_decay = ["bias", "LayerNorm.weight"]
    else:
        no_decay = []
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if Config().app.trainer.optimizer == 'SGD':
        return optim.SGD(optimizer_grouped_parameters,
                         lr=Config().app.trainer.learning_rate,
                         momentum=Config().app.trainer.momentum)
    elif Config().app.trainer.optimizer == 'Adam':
        return optim.Adam(optimizer_grouped_parameters,
                          lr=Config().app.trainer.learning_rate)
    elif Config().app.trainer.optimizer == 'AdamW':
        return optim.AdamW(optimizer_grouped_parameters,
                           lr=Config().app.trainer.learning_rate)

    raise ValueError('No such FL optimizer: {}'.format(
        Config().app.trainer.optimizer))


def get_loss_criterion():
    if hasattr(Config().app.trainer, "loss_criterion") \
            and Config().app.trainer.loss_criterion == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()
