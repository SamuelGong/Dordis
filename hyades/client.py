import asyncio
import logging
from hyades.clients import registry as client_registry
from hyades.config import Config


def run(client_id,
        port):
    Config().args.id = client_id
    if port is not None:
        Config().args.port = port
    else:
        client = client_registry.get()
        logging.info("Starting a %s client #%d.",
                     Config().clients.type, client_id)
        client.start_client()


if __name__ == "__main__":
    run(Config().args.id, Config().args.port)
