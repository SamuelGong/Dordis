import logging
from collections import OrderedDict

from hyades.apps.iterative import app as iterative_app
from hyades.apps.federated_learning \
    import app as federated_learning_app
from hyades.config import Config


registered_server = OrderedDict([
    ('iterative', iterative_app.AppServer),
    ('federated_learning', federated_learning_app.AppServer)
])

registered_clients = OrderedDict([
    ('iterative', iterative_app.AppClient),
    ('federated_learning', federated_learning_app.AppClient)
])


def get(client_id=None):
    app_type = Config().app.type

    if client_id is None:
        name = "Server application"
        registered_dict = registered_server
    else:
        name = "Client application"
        registered_dict = registered_clients

    if app_type in registered_dict:
        logging.info("%s: %s", name, app_type)
        registered_handler = registered_dict[app_type](client_id)
    else:
        raise ValueError(f"{name}: No such app handler: {app_type}")

    return registered_handler
