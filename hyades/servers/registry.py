import logging
from collections import OrderedDict

from hyades.servers import (
    base
)
from hyades.config import Config
from hyades.utils.github_related import git_status

registered_servers = OrderedDict([
    ('base', base.Server),
])


def get():
    server_type = Config().server.type

    if server_type in registered_servers:
        git_branch_name, git_head_hash_short = git_status()
        logging.info(
            f"Git: branch {git_branch_name}, commit {git_head_hash_short}")
        logging.info("Server: %s", server_type)
        registered_server = registered_servers[server_type]()
    else:
        raise ValueError('No such server: {}'.format(server_type))

    return registered_server
