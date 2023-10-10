import pickle

import numpy as np
import os
import time
import sys

import yaml

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_dir, '../../..'))
sys.path.append(os.path.join(file_dir, '../../analysis'))
from dordis.primitives.differential_privacy\
    .utils.accounting_utils import dskellam_params, skellam_epsilon

RDP_ORDERS = tuple(range(2, 129)) + (256,)

from exploration.analysis.plot_related import complex_plot

catelog_rel = "catelog.yml"
dropout_rate_list = np.arange(0, 0.5, 0.05)
method_list = ["Orig", "XNoise-Prec"]
plot_list = ["dropout-privacy"]
raw_data_file = "raw-data.pkl"

dropout_tolerance = 0.5


def get_params(app_type):
    if app_type == "femnist":
        target_epsilon = 6
        bits = 20
        # total_clients = 3550  # old
        total_clients = 1000
        clients_per_round = 100
        dim = 1018174
        seed = 1
        l2_clip_norm = 1
        # q = 0.02816901409  # old
        q = 1.0
        delta = 1. / total_clients
        beta = np.exp(-0.5)
        k_stddevs = 3
        steps = 50
        rho = 1
    elif app_type == "cifar10":
        target_epsilon = 6
        bits = 20
        total_clients = 100
        clients_per_round = 16
        dim = 11183562
        seed = 1
        l2_clip_norm = 3
        q = 1.0
        delta = 1. / total_clients
        beta = np.exp(-0.5)
        k_stddevs = 3
        steps = 150
        rho = 1
    elif app_type == "reddit":
        target_epsilon = 6
        bits = 20
        total_clients = 200
        clients_per_round = 20
        dim = 15091680
        seed = 1
        l2_clip_norm = 5
        q = 1.0
        delta = 1. / total_clients
        beta = np.exp(-0.5)
        k_stddevs = 3
        steps = 50
        rho = 1
    else:
        raise ValueError(f"Unknown app type: {app_type}.")

    return target_epsilon, bits, total_clients, clients_per_round, \
           dim, seed, l2_clip_norm, q, delta, beta, k_stddevs, steps, rho


def sum_std_upperbound(central_stddev, num_clients, dim,
                       scale, rho, sqrtn_norm_growth=False):
    n_factor = num_clients ** (1 if sqrtn_norm_growth else 2)
    part_1 = rho / dim * l2_clip_norm ** 2 * n_factor
    part_2 = 1 / 4.0 / scale ** 2 * num_clients
    part_3 = central_stddev ** 2
    return np.sqrt(part_1 + part_2 + part_3)

def main(args):
    start_time = time.perf_counter()

    app_type = args[0]
    target_epsilon, bits, total_clients, \
            clients_per_round, dim, seed, l2_clip_norm, \
            q, delta, beta, k_stddevs, steps, rho = get_params(app_type)

    np.random.seed(seed)
    padded_dim = np.math.pow(2, np.ceil(np.log2(dim)))

    print(f"Staring to generating data for {app_type}...")
    raw_data_path = os.path.join(file_dir, f"{app_type}_{raw_data_file}")
    if os.path.isfile(raw_data_path):
        with open(raw_data_path, 'rb') as fin:
            method_dict = pickle.load(fin)
    else:
        method_dict = {k: {} for k in method_list}
        for method in method_list:
            epsilon_list = []
            central_noise_list = []
            if method == "Early":
                for dropout_rate in dropout_rate_list:
                    epsilon_list.append(target_epsilon)
            elif method in ["Orig", "Con"]:
                if "dropout-privacy" in plot_list or "dropout-noise" in plot_list:
                    if method == "Orig":
                        target_num_clients = clients_per_round
                    else:  # Con
                        num_all_dropped = int(np.ceil(dropout_tolerance * clients_per_round))
                        target_num_clients = clients_per_round - num_all_dropped
                    scale, baseline_local_stddev = dskellam_params(
                        q=q,
                        epsilon=target_epsilon,
                        l2_clip_norm=l2_clip_norm,
                        bits=bits,
                        num_clients=target_num_clients,
                        dim=padded_dim,
                        delta=delta,
                        beta=beta,
                        steps=steps,
                        k=k_stddevs
                    )
                    print(f"DP parameters for {method} calculated.")

                    for dropout_rate in dropout_rate_list:
                        num_dropped = int(np.floor(dropout_rate * clients_per_round))
                        num_survived = clients_per_round - num_dropped
                        central_stddev = baseline_local_stddev \
                                         * np.sqrt(num_survived)

                        if "dropout-noise" in plot_list:
                            central_noise = central_stddev / l2_clip_norm
                            central_noise_list.append(central_noise)
                        if "dropout-privacy" in plot_list:
                            epsilon, _ = skellam_epsilon(
                                scale=scale,
                                central_stddev=central_stddev,
                                l2_sens=l2_clip_norm,
                                beta=beta,
                                dim=dim,
                                q=q,
                                steps=steps,
                                delta=delta,
                                orders=RDP_ORDERS
                            )
                            epsilon_list.append(epsilon)
            elif "XNoise" in method:
                if "dropout-privacy" in plot_list or "dropout-noise" in plot_list:
                    target_num_clients = clients_per_round
                    scale, baseline_local_stddev = dskellam_params(
                        q=q,
                        epsilon=target_epsilon,
                        l2_clip_norm=l2_clip_norm,
                        bits=bits,
                        num_clients=target_num_clients,
                        dim=padded_dim,
                        delta=delta,
                        beta=beta,
                        steps=steps,
                        k=k_stddevs
                    )
                    print(f"DP parameters for {method} calculated.")

                    num_all_dropped = int(np.ceil(dropout_tolerance * clients_per_round))
                    if method == "XNoise-Prec":
                        excessive_noise_component_stddev = [
                            baseline_local_stddev * np.sqrt(
                                clients_per_round / (clients_per_round - i) / (clients_per_round - i - 1)
                            ) for i in range(num_all_dropped)
                        ]
                    else:  # method == "XNoise-Appr"
                        noise_max_var = num_all_dropped / (clients_per_round - num_all_dropped) \
                                        * baseline_local_stddev ** 2
                        num_noise_levels = int(np.ceil(np.log2(num_all_dropped)))
                        noise_min_var = noise_max_var / 2 ** num_noise_levels

                        excessive_noise_component_stddev = []
                        for i in range(0, num_noise_levels):
                            excessive_noise_component_stddev.append(np.sqrt(2 ** i * noise_min_var))
                        excessive_noise_component_stddev = [excessive_noise_component_stddev[0]] \
                                                           + excessive_noise_component_stddev

                    for dropout_rate in dropout_rate_list:
                        num_dropped = int(np.floor(dropout_rate * clients_per_round))
                        num_survived = clients_per_round - num_dropped

                        if method == "XNoise-Prec":
                            local_stddev = np.sqrt((baseline_local_stddev ** 2
                                                    + sum([s ** 2 for s in
                                                           excessive_noise_component_stddev[:num_dropped]])))
                        else:  # method == "XNoise-Appr"
                            local_var_to_deduct = clients_per_round * baseline_local_stddev ** 2 * (
                                    1. / (clients_per_round - num_all_dropped) - 1. / (clients_per_round - num_dropped)
                            )
                            units_to_deduct = int(np.floor(local_var_to_deduct / noise_min_var))

                            component_idx_to_deduct = [0]
                            tmp = units_to_deduct - 1
                            for _idx in range(num_noise_levels):
                                if tmp & 1 == 1:
                                    component_idx_to_deduct.append(_idx + 1)
                                tmp >>= 1

                            component_idx_to_remain = []
                            for _idx in range(num_noise_levels + 1):
                                if _idx not in component_idx_to_deduct:
                                    component_idx_to_remain.append(_idx)

                            local_stddev = np.sqrt((baseline_local_stddev ** 2
                                                    + sum([excessive_noise_component_stddev[_idx] ** 2
                                                           for _idx in component_idx_to_remain])))

                        central_stddev = local_stddev \
                                         * np.sqrt(num_survived)

                        if "dropout-noise" in plot_list:
                            central_noise = central_stddev / l2_clip_norm
                            central_noise_list.append(central_noise)
                        if "dropout-privacy" in plot_list:
                            epsilon, _ = skellam_epsilon(
                                scale=scale,
                                central_stddev=central_stddev,
                                l2_sens=l2_clip_norm,
                                beta=beta,
                                dim=dim,
                                q=q,
                                steps=steps,
                                delta=delta,
                                orders=RDP_ORDERS
                            )
                            epsilon_list.append(epsilon)

                else:
                    raise NotImplementedError(f"Method {method} not supported.")

            if 'dropout-noise' in plot_list:
                method_dict[method].update({
                    'dropout-noise': {
                        'x': dropout_rate_list,
                        'y': central_noise_list
                    }
                })

            if 'dropout-privacy' in plot_list:
                method_dict[method].update({
                    'dropout-privacy': {
                        'x': dropout_rate_list,
                        'y': epsilon_list
                    }
                })
        with open(raw_data_path, 'wb') as fout:
            pickle.dump(method_dict, fout)
    print(method_dict)

    for plot_type in plot_list:
        if plot_type in ["dropout-privacy", "dropout-noise"]:
            x_label = "Dropout Rate (%)"
            # x_label = "Per-Round Dropout Rate (%)"
            if plot_type == "dropout-noise":
                # y_label = r"Central Noise Multiplier " \
                #           r"$\frac{\tilde{\sigma}}{\Delta_2}$"
                y_label = r"Central Noise Multiplier " \
                          r"$\frac{\sigma}{\Delta_2}$"
            else:
                y_label = r"Privacy $\epsilon$ ($\delta=" \
                          + f"{round(delta, 6)}" + "$)"
            data = []
            for method in method_list:
                if method == "XNoise-Prec":
                    method_name = "XNoise"
                else:
                    method_name = method
                data.append({
                    'label': method_name,
                    'x': method_dict[method][plot_type]['x'] * 100,  # percentage
                    'y': method_dict[method][plot_type]['y']
                })

            params = {
                "customized": {
                    'color_scheme': True,
                    "line_pattern": True,
                    'legend_separate': True
                },
                "legend": {
                    'ncol': 4,
                    'bbox_to_anchor': [1.2, 1.5],
                    "frameon": False,
                    "fontsize": 9
                },
                "figsize": [1.0, 1.1],
                "fontsize": 10,
                # "xticks": [0, 10, 20, 30, 40]
                "xticks": [0, 20, 40]
            }

            figure_file_path = os.path.join(file_dir, f'dps-{app_type}-' + plot_type)
            complex_plot(
                data=data,
                x_label=x_label,
                y_label=y_label,
                params=params,
                figure_file_name=figure_file_path,
                both_type=True
            )

    duration = round(time.perf_counter() - start_time, 2)
    print(f"[{duration}s] Plotted.")


if __name__ == "__main__":
    main(sys.argv[1:])
