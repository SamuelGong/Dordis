import argparse
import logging
import os
from collections import OrderedDict, namedtuple

import yaml


class Config:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('-i',
                                '--id',
                                type=str,
                                help='Unique client ID.')
            parser.add_argument('-p',
                                '--port',
                                type=str,
                                help='The port number for running a server.')
            parser.add_argument('-c',
                                '--config',
                                type=str,
                                default='./config.yml',
                                help='Federated learning configuration file.')
            parser.add_argument('-s',
                                '--server',
                                type=str,
                                default=None,
                                help='The server hostname and port number.')
            parser.add_argument('-l',
                                '--log',
                                type=str,
                                default='info',
                                help='Log messages level.')

            args = parser.parse_args()
            Config.args = args

            if Config.args.id is not None:
                Config.args.id = int(args.id)
            if Config.args.port is not None:
                Config.args.port = int(args.port)

            try:
                log_level = {
                    'critical': logging.CRITICAL,
                    'error': logging.ERROR,
                    'warn': logging.WARN,
                    'info': logging.INFO,
                    'debug': logging.DEBUG
                }[args.log]
            except KeyError:
                log_level = logging.INFO

            logging.basicConfig(
                format=
                '[%(levelname)s][%(asctime)s.%(msecs)03d] [%(filename)s:%(lineno)d]: %(message)s',
                level=log_level,
                datefmt='(%Y-%m-%d) %H:%M:%S')

            cls._instance = super(Config, cls).__new__(cls)

            if 'config_file' in os.environ:
                filename = os.environ['config_file']
            else:
                filename = args.config

            if os.path.isfile(filename):
                with open(filename, 'r') as config_file:
                    config = yaml.load(config_file, Loader=yaml.SafeLoader)
            else:
                config = Config.default_config()

            # to be backward compatible
            assert 'total_clients' in config['clients']

            Config.clients = Config.namedtuple_from_dict(config['clients'])
            Config.server = Config.namedtuple_from_dict(config['server'])
            Config.agg = Config.namedtuple_from_dict(config['agg'])
            Config.app = Config.namedtuple_from_dict(config['app'])
            Config.scheduler = Config.namedtuple_from_dict(config['scheduler'])

            if Config.args.server is not None:
                Config.server = Config.server._replace(
                    address=args.server.split(':')[0])
                Config.server = Config.server._replace(
                    port=args.server.split(':')[1])

            if 'simulation' in config:
                Config.simulation = Config.namedtuple_from_dict(config['simulation'])

            if 'results' in config:
                Config.results = Config.namedtuple_from_dict(config['results'])
                if hasattr(Config().results, 'results_dir'):
                    Config.result_dir = Config.results.results_dir
                else:
                    Config.result_dir = f'./results/'

            Config.params: dict = {}

            # A run ID is unique to each client in an experiment
            Config.params['run_id'] = os.getpid()

        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(typename='Config',
                                         field_names=fields,
                                         rename=True)
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields)
            try:
                return namedtuple_type(**field_value_pairs)
            except TypeError:
                return dict(**field_value_pairs)
        elif isinstance(obj, (list, set, tuple, frozenset)):
            return [Config.namedtuple_from_dict(item) for item in obj]
        else:
            return obj

    @staticmethod
    def default_config() -> dict:
        config = {}
        config['clients'] = {}
        config['clients']['total_clients'] = 10
        config['clients']['data'] = {}
        config['clients']['data']['source'] = 'random'
        config['clients']['data']['seed'] = 1
        config['clients']['data']['dim'] = 100
        config['clients']['data']['range'] = [-10.0, 10.0]
        config['server'] = {}
        config['server']['address'] = '127.0.0.1'
        config['server']['port'] = 8000
        config['server']['disable_clients'] = True
        config['agg'] = {}
        config['agg']['type'] = 'plaintext'
        config['agg']['threshold'] = 1.0
        config['app'] = {}
        config['app']['type'] = 'iterative'
        config['app']['init_scale_threshold'] = 1.0
        config['app']['repeat'] = 10
        config['scheduler'] = {}
        config['scheduler']["type"] = "base"

        return config

    @staticmethod
    def store() -> None:
        data = {}
        data['clients'] = Config.clients._asdict()
        data['server'] = Config.server._asdict()
        with open(Config.args.config, "w") as out:
            yaml.dump(data, out, default_flow_style=False)
