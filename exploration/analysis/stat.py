import os
import sys
from backend import extract_log
from text_related import text_summary
from plot_related import time_sequence_plot
sys.path.append('../..')
from functools import reduce
from hyades.config import Config
from hyades.apps.iterative.payload import Payload as iterative
from hyades.apps.federated_learning.payload import Payload as fl

config_rel = "config.yml"


def get_chunk_meta(task_folder):
    config_path = os.path.join(task_folder, config_rel)
    if os.path.isfile(config_path):
        os.environ["config_file"] = config_path

    if Config().app.type == "iterative":
        payload_model = iterative()
    else:
        # cannot be placed at the front due to import issues
        from hyades.apps.federated_learning.trainers \
            import registry as trainers_registry

        trainer = trainers_registry.get()
        payload_model = fl()
        payload_model.extract_model_meta(model=trainer.model)
    data_dim = payload_model.get_data_dim()

    # for setting DP parameters: TODO: to optmize, I think there is no need
    # to calculate DP parameters (it takes too long)
    from hyades.protocols import registry as protocol_registry
    protocol = protocol_registry.get()
    chunk_size = protocol.calc_chunk_size(data_dim, calc_dp_params=False)
    actual_dim = reduce(lambda x, y: x + y,
                        list(chunk_size.values()))
    padded_zeros_at_the_last_chunk = max(0, actual_dim - data_dim)

    Config._instance = None
    return chunk_size, data_dim, padded_zeros_at_the_last_chunk


def get_phase_info(task_folder):
    config_path = os.path.join(task_folder, config_rel)
    if os.path.isfile(config_path):
        os.environ["config_file"] = config_path
        agg_type = Config().agg.type

        if agg_type in ["plaintext", "secagg", "dp", "dp_plus_secagg"]:
            if agg_type in ["plaintext", "dp"]:
                from hyades.protocols.const import PlaintextConst
                c = PlaintextConst()

            else:  # secagg or dp_plus_secagg
                from hyades.protocols.const import SecAggConst
                c = SecAggConst()

            res = {}
            for key, value in c.__dict__.items():
                if "no_plot" not in key.lower():
                    res[value] = {
                        'name': key.lower(),
                        'type_idx': c._no_plot_phase_stage_mapping[value],
                        'chunk_relationship': c._no_plot_phase_mode_mapping[value]
                    }
        else:
            raise ValueError(f"Protocol type {agg_type} is not supported.")

        Config._instance = None
    else:
        raise ValueError(f"No {config_rel} in {task_folder}.")

    return res


def get_app_info(task_folder):
    config_path = os.path.join(task_folder, config_rel)
    res = {"test_metric": None, "is_fl": False}

    if os.path.isfile(config_path):
        os.environ["config_file"] = config_path
        if Config().app.type == "federated_learning":
            if hasattr(Config().app, "debug") \
                    and hasattr(Config().app.debug, "server") \
                    and hasattr(Config().app.debug.server, "test") \
                    and Config().app.debug.server.test:
                res.update({"test_metric": "accuracy"})
            res.update({"is_fl": True})
        elif Config().app.type == "iterative":
            if hasattr(Config().app, "debug") \
                    and hasattr(Config().app.debug, "server") \
                    and hasattr(Config().app.debug.server, "test") \
                    and Config().app.debug.server.test:
                res.update({"test_metric": "mse"})
    else:
        raise ValueError(f"No {config_rel} in {task_folder}.")

    Config._instance = None
    return res


def process_a_folder(task_folder):
    phase_info = get_phase_info(task_folder)
    chunk_meta = get_chunk_meta(task_folder)
    app_info = get_app_info(task_folder)

    result_dict, _ = extract_log(task_folder, phase_info,
                                 chunk_meta, app_info=app_info, mode="all")
    text_summary(task_folder, result_dict)

    time_sequence_plot(task_folder, result_dict, phase_info)


def main(args):
    task_folder = args[0]
    sys.argv.remove(task_folder)
    if not os.path.exists(task_folder):
        print(f'Folder {task_folder} does not exist.')
    elif os.path.isdir(task_folder):

        if config_rel in os.listdir(task_folder):
            process_a_folder(task_folder)
        else:
            # traverse all sub-folders if it is a parebnt folder
            parent_folder = task_folder
            for item in os.listdir(task_folder):
                task_folder = os.path.join(parent_folder, item)
                if os.path.isdir(task_folder):
                    process_a_folder(task_folder)


if __name__ == '__main__':
    main(sys.argv[1:])
