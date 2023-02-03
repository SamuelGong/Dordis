import ast
import copy
import os
import sys
import yaml
import json
import numpy as np
from backend import TIME_METRICS, NETWORK_METRICS, MSE, \
    RAW_DATA, COORDINATOR, ROUND, APP_METRICS, TESTING_ACCURACY
sys.path.append('../..')
from table_related import table_dim_3
from plot_related import complex_plot, bar_plot, stacked_bar_plot
from backend import find_substr

from hyades.primitives.differential_privacy\
    .utils.accounting_utils import dskellam_params, \
    skellam_rdp, get_privacy_spent

config_rel = "config.yml"
catelog_rel = "catelog.yml"
analysis_plan_rel = "analysis_plan.yml"
profile_plan_rel = "profile_plan.yml"
stat_json_rel = "stat.json"
METHOD = "methods"
CASE = "cases"
CLIENTS = "num_clients"
analysis_table_rel = "basic_tab.txt"
analysis_table_sa_rel = "basic_tab_sa.txt"
profile_table_rel = "profile_tab.txt"
coordinator_folder_rel = "hyades-coordinator"
log_rel = "log.txt"
RDP_ORDERS = tuple(range(2, 129)) + (256,)


def preprocess_analysis_plan(plan_path):
    with open(plan_path, 'r') as fin:
        plan = yaml.load(fin, Loader=yaml.FullLoader)

    data = {}
    for method in plan[METHOD]:
        data[method] = {}
        for num_clients in plan[CLIENTS]:
            data[method][num_clients] = {}
            for case in plan[CASE]:
                data[method][num_clients][str(case)] = {}

    if "params" in plan:
        params = plan["params"]
    else:
        params = {}

    return data, params


def preprocess_profile_plan(plan_path):
    with open(plan_path, 'r') as fin:
        plan = yaml.load(fin, Loader=yaml.FullLoader)

    data = {}
    for case in plan[CASE]:
        data[str(case)] = {}

    if "params" in plan:
        params = plan["params"]
    else:
        params = {}

    return data, params


def extract_app_metric(task_folder, metric):
    stat_path = os.path.join(task_folder, stat_json_rel)
    with open(stat_path, 'r') as fin:
        result_dict = json.load(fin)

    res = {}
    coordinator_dict = result_dict[RAW_DATA][COORDINATOR][ROUND]
    for round_idx, round_dict in coordinator_dict.items():
        if APP_METRICS in round_dict:
            res[round_idx] = round_dict[APP_METRICS][metric]

    # check if the time is simulated
    config_path = os.path.join(task_folder, config_rel)
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    if "simulation" in config and "time" in config["simulation"]:
        log_path = os.path.join(task_folder, coordinator_folder_rel, log_rel)
        with open(log_path, 'r') as fin:
            lines = fin.readlines()

        new_round_latency_dict = {}
        total_time = 0
        for line in lines:
            if "Stats update" in line:
                lat_str = ":".join(find_substr(
                    line=line,
                    preceding_string="Stats update for Round ",
                    succeeding_separator="}."
                ).split(':')[1:])[1:] + "}"
                lat_d = ast.literal_eval(lat_str)
                lat_max = -1
                for _, v in lat_d.items():
                    if lat_max < v['time']:
                        lat_max = v['time']

                total_time += lat_max

                round_str = find_substr(
                    line=line,
                    preceding_string="[Round ",
                    succeeding_separator="]"
                )
                new_round_latency_dict[round_str] = total_time

        for round_str, round_dict in res.items():
            res[round_str]['time'] = new_round_latency_dict[round_str]

    return res


def extract_time_info(task_folder, part='all'):
    stat_path = os.path.join(task_folder, stat_json_rel)
    with open(stat_path, 'r') as fin:
        result_dict = json.load(fin)

    return {
        'value': result_dict[TIME_METRICS][part]['mean'],
        'std': result_dict[TIME_METRICS][part]['std']
    }


def extract_dropout_info(task_folder):
    log_path = os.path.join(
        task_folder, coordinator_folder_rel, log_rel
    )
    with open(log_path, 'r') as fin:
        lines = fin.readlines()

    config_path = os.path.join(
        task_folder, config_rel
    )
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    if "differential_privacy" in config["agg"] \
            and "dropout_resilience" in config["agg"]["differential_privacy"]:
        dropout_tolerated_frac = config["agg"]["differential_privacy"][
            "dropout_resilience"]["dropout_tolerated_frac"]
    else:  # placeholder
        dropout_tolerated_frac = 0

    num_sampled_clients_list = []
    num_dropped_clients_list = []
    dropout_tolerance_list = []
    for line in lines:
        if "Sampled clients" in line:
            sampled_clients = find_substr(
                line=line,
                preceding_string="Sampled clients (",
                succeeding_separator=")",
                mode="int"
            )
            dropout_tolerated = int(np.floor(
                sampled_clients * dropout_tolerated_frac))
            num_sampled_clients_list.append(sampled_clients)
            dropout_tolerance_list.append(dropout_tolerated)
        elif "U2/U3" in line:  # TODO: avoid hard-coding
            dropped_clients_literal = find_substr(
                line=line,
                preceding_string="U2/U3: ",
                succeeding_separator=".\n",
                mode="str"
            )
            dropped_clients = ast.literal_eval(dropped_clients_literal)
            num_dropped_clients_list.append(len(dropped_clients))

    return {
        "num_sampled_clients": num_sampled_clients_list,
        "num_dropped_clients": num_dropped_clients_list,
        "dropout_tolerance": dropout_tolerance_list
    }


def extract_privacy_params(task_folder):
    log_path = os.path.join(task_folder, coordinator_folder_rel, log_rel)
    with open(log_path, 'r') as fin:
        lines = fin.readlines()

    to_collect = {
        "epsilon": "float",
        "delta": "float",
        "bits": "int",
        "beta": "float",
        "l2_clip_norm": "float",
        "k_stddevs": "int",
        "target_num_clients": "int",
        "client_sampling_rate": "float",
        "dim": "int",
        "padded_dim": "float",
        "gamma": "float",
        "scale": "float",
        "local_stddev": "float",
        "local_scale": "float"
    }
    result = {}
    for line in lines:
        if "Initialized parameters for" in line:
            for item, mode in to_collect.items():
                # TODO: avoid hard-coding
                succeeding_separate = ","
                # if item == "local_scale":  # as it is the last one
                #     succeeding_separate = "}"

                tmp = find_substr(
                    line=line,
                    preceding_string=f"'{item}': ",
                    succeeding_separator=succeeding_separate,
                    mode=mode
                )
                result[item] = tmp
            break

    # if pessimistic
    config_path = os.path.join(
        task_folder, config_rel
    )
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    if "differential_privacy" in config["agg"] \
            and "pessimistic" in config["agg"]["differential_privacy"]:
        pessimistic_dropout_tolerated_frac \
            = config["agg"]["differential_privacy"]["pessimistic"]
        result["pessimistic_dropout_tolerated_frac"] \
            = pessimistic_dropout_tolerated_frac

    return result


def calc_privacy_loss_accumulated(method, dropout_info,
                                  privacy_params, verbose=True):
    num_sampled_clients = dropout_info["num_sampled_clients"]
    num_dropped_clients = dropout_info["num_dropped_clients"]
    dropout_tolerance = dropout_info["dropout_tolerance"]

    target_epsilon = privacy_params["epsilon"]
    target_delta = privacy_params["delta"]
    scale = privacy_params["scale"]
    precomputed_local_stddev = privacy_params["local_stddev"]
    l2_clip_norm = privacy_params["l2_clip_norm"]
    beta = privacy_params["beta"]
    padded_dim = privacy_params["padded_dim"]
    sampling_rate_upperbound = privacy_params["client_sampling_rate"]
    # currently, target_num_clients = num_sampled_clients_upperbound
    # (e.g., for trace-driven sampler, it is int(np.floor(
    #     self.total_clients * self.sampling_rate_upperbound
    # ))
    target_num_clients = privacy_params["target_num_clients"]

    rdp_epsilon_sum = None
    privacy_spent_list = []
    round_cnt = 0
    actual_central_stddev_list = []

    for num_sampled_client, num_dropped_client, dropout_tolerated in zip(
            num_sampled_clients, num_dropped_clients, dropout_tolerance):
        required_central_variance = precomputed_local_stddev ** 2 * target_num_clients
        if method in ["Orig", "Early"]:
            actual_central_stddev = np.sqrt(
                required_central_variance * (
                    1 - num_dropped_client / num_sampled_client
                )
            )
        elif method == "Con":
            pessimistic_dropout_tolerated_frac \
                = privacy_params["pessimistic_dropout_tolerated_frac"]
            actual_target_num_clients = num_sampled_client
            actual_target_num_clients *= (1 - pessimistic_dropout_tolerated_frac)
            actual_target_num_clients = max(1, int(np.floor(actual_target_num_clients)))

            actual_central_stddev = np.sqrt(
                required_central_variance * (
                    (num_sampled_client - num_dropped_client)
                    / actual_target_num_clients
                )
            )
        else:
            per_round_baseline_local_stddev = np.sqrt(required_central_variance / num_sampled_client)
            # print(num_sampled_client, num_dropped_client, dropout_tolerated)
            # print(required_central_variance, per_round_baseline_local_stddev)
            if method == "XNoise-Prec":
                excessive_noise_component_stddev = [
                    per_round_baseline_local_stddev * np.sqrt(
                        num_sampled_client / (num_sampled_client - i)
                        / (num_sampled_client - i - 1)
                    ) for i in range(dropout_tolerated)
                ]
                new_local_stddev = np.sqrt((per_round_baseline_local_stddev ** 2
                                        + sum([s ** 2 for s in
                                               excessive_noise_component_stddev[
                                               :num_dropped_client]])))
            else:  # XNoise-Appr
                noise_max_var = dropout_tolerated / (num_sampled_client - dropout_tolerated) \
                                * per_round_baseline_local_stddev ** 2
                num_noise_levels = int(np.ceil(np.log2(dropout_tolerated)))
                noise_min_var = noise_max_var / 2 ** num_noise_levels

                excessive_noise_component_stddev = []
                for i in range(0, num_noise_levels):
                    excessive_noise_component_stddev.append(np.sqrt(2 ** i * noise_min_var))
                excessive_noise_component_stddev = [excessive_noise_component_stddev[0]] \
                                                   + excessive_noise_component_stddev

                local_var_to_deduct = num_sampled_client * per_round_baseline_local_stddev ** 2 * (
                        1. / (num_sampled_client - dropout_tolerated)
                        - 1. / (num_sampled_client - num_dropped_client)
                )
                units_to_deduct = int(np.floor(local_var_to_deduct / noise_min_var))
                # print(noise_max_var, noise_min_var)
                # print([e ** 2 for e in excessive_noise_component_stddev])

                if units_to_deduct > 0:
                    component_idx_to_deduct = [0]
                    tmp = units_to_deduct - 1
                    for _idx in range(num_noise_levels):
                        if tmp & 1 == 1:
                            component_idx_to_deduct.append(_idx + 1)
                        tmp >>= 1
                else:
                    component_idx_to_deduct = []

                # print(local_var_to_deduct, units_to_deduct, component_idx_to_deduct)
                # print('---')

                component_idx_to_remain = []
                for _idx in range(num_noise_levels + 1):
                    if _idx not in component_idx_to_deduct:
                        component_idx_to_remain.append(_idx)

                new_local_stddev = np.sqrt((per_round_baseline_local_stddev ** 2
                                        + sum([excessive_noise_component_stddev[_idx] ** 2
                                               for _idx in component_idx_to_remain])))

            actual_central_stddev = new_local_stddev \
                             * np.sqrt(num_sampled_client
                                       - num_dropped_client)  # num_survived

        actual_central_stddev_list.append(actual_central_stddev)
        rdp_epsilon = skellam_rdp(
            scale=scale,
            central_stddev=actual_central_stddev,
            l2_sens=l2_clip_norm,
            beta=beta,
            dim=padded_dim,
            q=sampling_rate_upperbound,
            steps=1,
            orders=RDP_ORDERS
        )
        rdp_epsilon = np.array(rdp_epsilon)
        if rdp_epsilon_sum is None:
            rdp_epsilon_sum = rdp_epsilon
        else:
            rdp_epsilon_sum += rdp_epsilon

        # print(round_cnt, central_stddev)  # for inspecting the speed
        actual_epsilon, _, _ = get_privacy_spent(
            RDP_ORDERS, rdp_epsilon_sum, target_delta=target_delta
        )
        privacy_spent_list.append(actual_epsilon)
        round_cnt += 1

    if verbose:
        print(f"\tStatistics of actual_central_stddev: "
              f"max {max(actual_central_stddev_list)}, "
              f"mean {np.mean(actual_central_stddev_list)}, "
              f"min {min(actual_central_stddev_list)}.")

    return privacy_spent_list, target_epsilon



def get_truncate_idx(arr, threshold):
    truncate_idx = len(arr)
    while truncate_idx > 0:
        used_epsilon = round(arr[truncate_idx - 1],
                             6)  # to ignore small error
        if used_epsilon <= threshold:
            break
        truncate_idx -= 1

    return truncate_idx


def app_metric_for_early_stop(folder_path, app_metric, verbose=True):
    dropout_info = extract_dropout_info(
        task_folder=folder_path
    )
    privacy_params = extract_privacy_params(
        task_folder=folder_path
    )
    privacy_spent_list, target_epsilon = calc_privacy_loss_accumulated(
        method="Early",
        dropout_info=dropout_info,
        privacy_params=privacy_params,
        verbose=verbose
    )
    # to ignore small errors
    truncate_idx = get_truncate_idx(
        arr=privacy_spent_list,
        threshold=target_epsilon
    )
    # print(privacy_spent_list)
    if verbose:
        print(truncate_idx)

    new_app_metrics = {}
    for round_str in app_metric.keys():
        round_idx = int(round_str)
        if round_idx < truncate_idx:
            new_app_metrics[round_idx] = app_metric[round_str]

    return new_app_metrics


def get_folder_path(parent_folder, exp_name):
    # find key by value
    catelog_path = os.path.join(parent_folder, catelog_rel)
    with open(catelog_path, 'r') as fin:
        catelog = yaml.load(fin, Loader=yaml.FullLoader)
    folder_name = list(catelog.keys())[list(catelog.values()).index(exp_name)]
    return os.path.join(parent_folder, folder_name)


def extract_exp_data(method_name, folder_path, plot_types, verbose=True):
    result = {}
    for plot_type in plot_types:
        if plot_type == "time-stacked-bar":
            overall_time = extract_time_info(folder_path)["value"]
            agg_time = extract_time_info(folder_path, part='agg')["value"]
            result[plot_type] = {
                'agg' : agg_time,
                'other': overall_time - agg_time
            }

        elif "round-acc" in plot_types:
            app_metric = extract_app_metric(folder_path, metric=TESTING_ACCURACY)
            if method_name == "Early":
                app_metric = app_metric_for_early_stop(
                    folder_path=folder_path,
                    app_metric=app_metric,
                    verbose=verbose
                )

            round_idx_list = list(app_metric.keys())
            acc_list = [app_metric[round_idx]["value"] for round_idx in round_idx_list]

            result[plot_type] = {
                'x': round_idx_list,
                'y': acc_list
            }
        elif "time-acc" in plot_types:
            app_metric = extract_app_metric(folder_path, metric=TESTING_ACCURACY)
            if method_name == "Early":
                app_metric = app_metric_for_early_stop(
                    folder_path=folder_path,
                    app_metric=app_metric,
                    verbose=verbose
                )

            round_idx_list = list(app_metric.keys())
            time_list = [app_metric[round_idx]["time"] for round_idx in round_idx_list]
            acc_list = [app_metric[round_idx]["value"] for round_idx in round_idx_list]
            result[plot_type] = {
                'x': time_list,
                'y': acc_list
            }

    return result


def batch_plot(parent_folder, plot_plan_path):
    # step 1: extract data
    with open(plot_plan_path, 'r') as fin:
        plan = yaml.load(fin, Loader=yaml.FullLoader)

    assert "plots" in plan
    plot_types = []
    for plot_dict in plan["plots"]:
        plot_types.append(plot_dict["type"])

    assert "enable" in plan

    raw_data = {}
    for group in plan["enable"]:
        name = group["group"]
        member_data = group["member"]
        if name not in raw_data:
            raw_data[name] = {}

        for exp_data in member_data:
            label = exp_data["label"]
            relative_folder = exp_data["folder"]
            exp_name = exp_data["exp_name"]

            folder_path = get_folder_path(
                os.path.join(parent_folder, relative_folder),
                exp_name
            )
            _data = extract_exp_data(
                method_name=name,
                folder_path=folder_path,
                plot_types=plot_types
            )

            raw_data[name][label] = _data

    for plot_dict in plan["plots"]:
        plot_type = plot_dict["type"]
        xlabel = plot_dict["xlabel"] if "xlabel" in plot_dict else None
        ylabel = plot_dict["ylabel"] if "ylabel" in plot_dict else None

        if plot_type == "time-stacked-bar":
            data_to_plot = {}
            for outer_key, outer_value in raw_data.items():
                if outer_key not in data_to_plot:
                    data_to_plot[outer_key] = {}
                for inner_key, inner_value in outer_value.items():
                    data_to_plot[outer_key][inner_key] = inner_value[plot_type]

            stacked_bar_plot(
                data=data_to_plot,
                x_label=xlabel,
                y_label=ylabel,
                figure_file_name=os.path.join(
                    parent_folder, plot_dict["name"]),
                params=plot_dict["params"],
                both_type=True,
            )
        elif plot_type == "round-acc":
            data_to_plot = []
            ylabel += " (%)"

            for group_name, group_data in raw_data.items():
                # there should be only 1 key
                label = group_name
                first_key = list(group_data)[0]
                label_data = group_data[first_key][plot_type]

                x = np.array([int(e) for e in label_data['x']]) + 1
                y = np.array(label_data['y']) * 100  # to percent

                data_to_plot.append({
                    'label': label,
                    'x': x,
                    'y': y
                })

            complex_plot(
                data=data_to_plot,
                x_label=xlabel,
                y_label=ylabel,
                figure_file_name=os.path.join(
                    parent_folder, plot_dict["name"]),
                params=plot_dict["params"],
                both_type=True,
                # preprocess_acc=False,
                preprocess_acc = True
            )

        elif plot_type == "time-acc":
            data_to_plot = []
            ylabel += " (%)"

            max_second = -1
            for group_name, group_data in raw_data.items():
                # there should be only 1 key
                label = group_name
                first_key = list(group_data)[0]
                label_data = group_data[first_key][plot_type]

                x = np.array(label_data['x'])
                if max_second < max(x):
                    max_second = max(x)
                y = np.array(label_data['y']) * 100  # to percent

                data_to_plot.append({
                    'label': label,
                    'x': x,
                    'y': y
                })

            if max_second <= 120:
                xlabel += " (s)"
            elif 120 < max_second <= 7200:
                xlabel += " (min)"
                for label, item in enumerate(data_to_plot):
                    data_to_plot[label]['x'] = item['x'] / 60
            else:
                xlabel += " (h)"
                for label, item in enumerate(data_to_plot):
                    data_to_plot[label]['x'] = item['x'] / 3600

            complex_plot(
                data=data_to_plot,
                x_label=xlabel,
                y_label=ylabel,
                figure_file_name=os.path.join(
                    parent_folder, plot_dict["name"]),
                params=plot_dict["params"],
                both_type=True,
                preprocess_acc=True
            )

    # if "enable-2" not in plan:
    #     data = extract_exp_data(plan, parent_folder, plot_types)
    # else:
    #     data = {}
    #     for folder in plan['enable-2']:
    #         label = folder["label"]
    #         postfix = folder["folder_name"]
    #         _data = extract_exp_data(
    #             plan=plan,
    #             parent_folder=os.path.join(parent_folder, postfix),
    #             plot_types=plot_types
    #         )
    #         data[label] = _data
    #
    # # print(data)
    # # step 2: plot
    # for plot_dict in plan["plots"]:
    #     type = plot_dict["type"]
    #     xlabel = plot_dict["xlabel"] if "xlabel" in plot_dict else None
    #     ylabel = plot_dict["ylabel"] if "ylabel" in plot_dict else None
    #
    #     if type in ["time-to-acc", "round-to-acc", "round-acc"]:
    #         data_to_plot = []
    #         ylabel += " (%)"
    #         if type == "time-to-acc":
    #             max_second = -1
    #         for label, label_data in data.items():
    #             if type == "time-to-acc":
    #                 x = np.array(label_data[type]['x'])
    #                 y = np.array(label_data[type][type]) * 100
    #             else: # round-to-acc or round-acc
    #                 x = np.array([int(e) for e in label_data["round-to-acc"]['x']]) + 1
    #                 y = np.array(label_data["round-to-acc"]['y']) * 100
    #
    #             if type == "time-to-acc":
    #                 m = max(x)
    #                 if m > max_second:
    #                     max_second = m
    #
    #             data_to_plot.append({
    #                 'label': label,
    #                 'x': x,
    #                 'y': y
    #             })
    #
    #         if type == "time-to-acc":
    #             if max_second <= 120:
    #                 xlabel += " (s)"
    #             elif 120 < max_second <= 7200:
    #                 xlabel += " (min)"
    #                 for label, item in enumerate(data_to_plot):
    #                     data_to_plot[label]['x'] = item['x'] / 60
    #             else:
    #                 xlabel += " (h)"
    #                 for label, item in enumerate(data_to_plot):
    #                     data_to_plot[label]['x'] = item['x'] / 3600
    #
    #         complex_plot(
    #             data=data_to_plot,
    #             x_label=xlabel,
    #             y_label=ylabel,
    #             figure_file_name=os.path.join(
    #                 parent_folder, plot_dict["name"]),
    #             params=plot_dict["params"],
    #             both_type=True,
    #             preprocess_acc=True
    #         )
    #     elif type in ["round-time"]:
    #         data_to_plot = {}
    #
    #         if type == "round-time":
    #             max_second = -1
    #
    #         for label, label_data in data.items():
    #             m = max(label_data)
    #             if m > max_second:
    #                 max_second = m
    #             data_to_plot[label] = label_data[type]
    #
    #         if type == "round-time":
    #             if max_second <= 120:
    #                 ylabel += " (s)"
    #             elif 120 < max_second <= 7200:
    #                 ylabel += " (min)"
    #                 for label, item in data_to_plot.items():
    #                     data_to_plot[label]['value'] = item['value'] / 60
    #                     data_to_plot[label]['std'] = item['std'] / 60
    #             else:
    #                 ylabel += " (h)"
    #                 for label, item in data_to_plot.items():
    #                     data_to_plot[label]['value'] = item['value'] / 3600
    #                     data_to_plot[label]['std'] = item['std'] / 3600
    #
    #         bar_plot(
    #             x_label=xlabel,
    #             y_label=ylabel,
    #             data=data_to_plot,
    #             figure_file_name=os.path.join(
    #                 parent_folder, plot_dict["name"]),
    #             params=plot_dict["params"],
    #             both_type=True
    #         )
    #     elif type in ["dropout-mse", "dropout-acc"]:
    #         if type == "dropout-mse":
    #             ylabel += r' $\frac{\Vert \^{x} - x \Vert_2^2}{d}$'
    #
    #         # bar plot
    #         # data_to_plot = {}
    #         # for label, label_data in data.items():
    #         #     label /= 100
    #         #     data_to_plot[label] = label_data[type]
    #         # bar_plot(
    #         #     x_label=xlabel,
    #         #     y_label=ylabel,
    #         #     data=data_to_plot,
    #         #     figure_file_name=os.path.join(
    #         #         parent_folder, plot_dict["name"]),
    #         #     params=plot_dict["params"],
    #         #     both_type=True
    #         # )
    #
    #         data_to_plot = []
    #         if type not in data[list(data.keys())[0]]:
    #             # it is a 2-level dict
    #             for label, d in data.items():
    #                 _data = {
    #                     'x': [],
    #                     'y': [],
    #                     'y_std': [],
    #                     'label': label
    #                 }
    #                 if type == "dropout-mse":
    #                     for dropout_rate, mean_mse in d.items():
    #                         dropout_rate /= 100
    #                         _data['x'].append(dropout_rate)
    #                         _data['y'].append(mean_mse[type]["value"])
    #                         _data['y_std'].append(mean_mse[type]["std"])
    #                 else:
    #                     for dropout_rate, mean_acc in d.items():
    #                         dropout_rate /= 100
    #                         _data['x'].append(dropout_rate)
    #                         _data['y'].append(mean_acc[type]["value"])
    #                         # _data['y_std'].append(mean_acc[type]["std"])
    #                 data_to_plot.append(_data)
    #         else:
    #             _data = {
    #                 'x': [],
    #                 'y': [],
    #                 'y_std': []
    #             }
    #             if type == "dropout-mse":
    #                 for dropout_rate, mean_mse in data.items():
    #                     dropout_rate /= 100
    #                     _data['x'].append(dropout_rate)
    #                     _data['y'].append(mean_mse[type]["value"])
    #                     _data['y_std'].append(mean_mse[type]["std"])
    #             else:
    #                 for dropout_rate, mean_acc in data.items():
    #                     dropout_rate /= 100
    #                     _data['x'].append(dropout_rate)
    #                     _data['y'].append(mean_acc[type]["value"])
    #                     # _data['y_std'].append(mean_acc[type]["std"])
    #             data_to_plot.append(_data)
    #
    #         complex_plot(
    #             x_label=xlabel,
    #             y_label=ylabel,
    #             data=data_to_plot,
    #             figure_file_name=os.path.join(
    #                 parent_folder, plot_dict["name"]),
    #             params=plot_dict["params"],
    #             both_type=True
    #         )


def recover_exp_name(filename):
    l = filename.split('_')
    if len(l) == 5:
        algo, case, num_clients, para, pipe = l
        return "-".join([algo, para, pipe]), case, int(num_clients)
    elif len(l) == 6:
        algo, case, num_clients, para, pipe, dropout_rate = l
        return "-".join([algo, para, f"{pipe}-{dropout_rate}"]), \
               case, int(num_clients)


def basic_analysis(parent_folder, catelog, data, params=None):
    data_sa_only = copy.deepcopy(data)
    for folder in catelog.keys():
        task_folder = os.path.join(parent_folder, folder)
        if not os.path.isdir(task_folder):
            print(f"Folder {task_folder} does not exist.")
            continue

        exp_id = recover_exp_name(catelog[folder])
        method, case, num_clients = exp_id
        if method not in data \
                or num_clients not in data[method] \
                or case not in data[method][num_clients]:
            continue

        stat_path = os.path.join(task_folder, stat_json_rel)
        if os.path.isfile(stat_path):
            with open(stat_path, 'r') as fin:
                result_dict = json.load(fin)

            duration = result_dict[TIME_METRICS]["all"]["mean"]
            traffic = result_dict[NETWORK_METRICS]["total"]["mean"]
            agg_duration = result_dict[TIME_METRICS]["agg"]["mean"]
            agg_traffic = result_dict[NETWORK_METRICS]["agg_total"]["mean"]

            data[method][num_clients][case].update({
                "duration": round(duration, 1),
                "traffic": round(traffic, 2)
            })

            data_sa_only[method][num_clients][case].update({
                "duration": round(agg_duration, 1),
                "traffic": round(agg_traffic, 2),
            })

    print('End-to-end:')
    table_dim_3(parent_folder, data, analysis_table_rel, params)
    print('SA only:')
    table_dim_3(parent_folder, data_sa_only, analysis_table_sa_rel, params)


def basic_profile(parent_folder, catelog, data, params=None):
    for folder in catelog.keys():
        task_folder = os.path.join(parent_folder, folder)
        if not os.path.isdir(task_folder):
            print(f"Folder {task_folder} does not exist.")
            continue

        exp_id = recover_exp_name(catelog[folder])
        _, case, num_clients = exp_id
        if case not in data:
            continue

        if os.path.isdir(task_folder):
            stat_path = os.path.join(task_folder, stat_json_rel)
            if not os.path.isfile(stat_path):
                print(f"Folder {task_folder} does not exist.")
                continue

            with open(stat_path, 'r') as fin:
                stat = json.load(fin)
                stat = stat["Raw data"]["average_view"]

                for phase_idx, phase_dict in stat.items():
                    if phase_idx not in data[case]:
                        data[case][phase_idx] = {}
                    data[case][phase_idx] = \
                        {num_clients: phase_dict}

    table_dim_3(parent_folder, data, profile_table_rel, params)


def main(args):
    parent_folder = args[0]

    plot_plan_rel = None
    if len(args) > 1:
        plot_plan_rel = args[1]

    if plot_plan_rel is None:
        plot_plan_rel = "plot_plan.yml"

    sys.argv.remove(parent_folder)
    if not os.path.exists(parent_folder):
        print(f'Folder {parent_folder} does not exist.')
    elif os.path.isdir(parent_folder):
        # should be a parent folder of tasks
        assert config_rel not in os.listdir(parent_folder)
        catelog_path = os.path.join(parent_folder, catelog_rel)
        with open(catelog_path, 'r') as fin:
            catelog = yaml.load(fin, Loader=yaml.FullLoader)

        # basic analysis
        analysis_plan_path = os.path.join(parent_folder, analysis_plan_rel)
        if os.path.isfile(analysis_plan_path):
            data, params = preprocess_analysis_plan(analysis_plan_path)

            basic_tab_params = None
            if "basic_tab" in params:
                basic_tab_params = params["basic_tab"]
            basic_analysis(
                parent_folder=parent_folder,
                catelog=catelog,
                data=data,
                params=basic_tab_params
            )

        # stage duration profiling
        profile_plan_path = os.path.join(parent_folder, profile_plan_rel)
        if os.path.isfile(profile_plan_path):
            data, params = preprocess_profile_plan(profile_plan_path)
            basic_profile(
                parent_folder=parent_folder,
                catelog=catelog,
                data=data,
                params=params
            )

        # plotting
        plot_plan_path = os.path.join(parent_folder, plot_plan_rel)
        if os.path.isfile(plot_plan_path):
            batch_plot(
                parent_folder=parent_folder,
                plot_plan_path=plot_plan_path
            )
    else:
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
