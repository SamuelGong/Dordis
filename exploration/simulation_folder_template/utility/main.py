import pickle
import time
import ast
import numpy as np
import os
import sys
import json
import yaml

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_dir, '../../..'))
sys.path.append(os.path.join(file_dir, '../../analysis'))
from dordis.primitives.differential_privacy\
    .utils.accounting_utils import dskellam_params, \
    skellam_rdp, get_privacy_spent
from plot_related import complex_plot, stacked_bar_plot
from backend import find_substr


seed = 1
np.random.seed(seed)
catelog_rel = "catelog.yml"
# stat_json_rel = "stat.json"
aggregator_folder_rel = 'dordis-coordinator'
log_rel = "log.txt"
config_rel = "config.yml"

used_acc_metric = "last-10-avg"
# used_acc_metric = "last-1-avg"
RDP_ORDERS = tuple(range(2, 129)) + (256,)


def basic_acc_info(acc_list):
    return {
        'raw': acc_list,
        'count': len(acc_list),
        'max': max(acc_list),
        'min': min(acc_list),
        'argmax': np.argmax(acc_list),
        'last-1-avg': np.mean(acc_list[-1:]),
        'last-1-std': np.std(acc_list[-1:]),
        'last-3-avg': np.mean(acc_list[-3:]),
        'last-3-std': np.std(acc_list[-3:]),
        'last-10-avg': np.mean(acc_list[-10:]),
        'last-10-std': np.std(acc_list[-10:]),
        'last-20-avg': np.mean(acc_list[-20:]),
        'last-20-std': np.std(acc_list[-30:]),
        'last-30-avg': np.mean(acc_list[-30:]),
        'last-30-std': np.std(acc_list[-30:]),
    }


def extract_acc_info(task_folder):
    log_path = os.path.join(task_folder, aggregator_folder_rel, log_rel)
    with open(log_path, 'r') as fin:
        lines = fin.readlines()

    acc_list = []
    for line in lines:
        if "Testing accuracy" in line:
            acc = find_substr(
                line=line,
                preceding_string="Testing accuracy: ",
                succeeding_separator=".\n",
                mode="float"
            )
            acc_list.append(acc)

    return basic_acc_info(acc_list)


def extract_dropout_info(task_folder):
    log_path = os.path.join(
        task_folder, aggregator_folder_rel, log_rel
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
    log_path = os.path.join(task_folder, aggregator_folder_rel, log_rel)
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

                if item == "steps":  # as it is the last one
                    succeeding_separate = "}"

                tmp = find_substr(
                    line=line,
                    preceding_string=f"'{item}': ",
                    succeeding_separator=succeeding_separate,
                    mode=mode
                )
                result[item] = tmp
            break

    return result


def calc_privacy_loss_accumulated(method, dropout_info, privacy_params):
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

    for num_sampled_client, num_dropped_client, dropout_tolerated in zip(
            num_sampled_clients, num_dropped_clients, dropout_tolerance):
        required_central_variance = precomputed_local_stddev ** 2 * target_num_clients
        if method in ["Orig", "Early"]:
            actual_central_stddev = np.sqrt(
                required_central_variance * (
                    1 - num_dropped_client / num_sampled_client
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


def acc_info_for_early_stop(folder_path, acc_info):
    dropout_info = extract_dropout_info(
        task_folder=folder_path
    )
    privacy_params = extract_privacy_params(
        task_folder=folder_path
    )
    privacy_spent_list, target_epsilon = calc_privacy_loss_accumulated(
        method="Early",
        dropout_info=dropout_info,
        privacy_params=privacy_params
    )
    # to ignore small errors
    truncate_idx = get_truncate_idx(
        arr=privacy_spent_list,
        threshold=target_epsilon
    )
    print(privacy_spent_list)
    print(truncate_idx)

    acc_list = acc_info["raw"]
    acc_list = acc_list[:truncate_idx]
    acc_info = basic_acc_info(acc_list)

    return acc_info


def extract_data(group_name, member_name, exp_name, parent_folder, plots):
    catelog_path = os.path.join(parent_folder, catelog_rel)
    with open(catelog_path, 'r') as fin:
        catelog = yaml.load(fin, Loader=yaml.FullLoader)

    # find key by (the last matching) value
    catelog_value = list(catelog.values())
    catelog_key = list(catelog.keys())
    folder_name = catelog_key[catelog_value.index(exp_name)]
    folder_path = os.path.join(parent_folder, folder_name)

    result = {}
    for plot_dict in plots:
        type = plot_dict["type"]
        if type in ["dropout-acc", "dropout-acc-line"]:
            acc_info = extract_acc_info(
                task_folder=folder_path
            )

            if member_name == "Early":
                acc_info = acc_info_for_early_stop(
                    folder_path=folder_path,
                    acc_info=acc_info
                )
            result["acc_info"] = acc_info

        elif type in ["dropout-privacy"]:
            dropout_info = extract_dropout_info(
                task_folder=folder_path
            )
            privacy_params = extract_privacy_params(
                task_folder=folder_path
            )
            privacy_spent_list, target_epsilon = calc_privacy_loss_accumulated(
                method=member_name,
                dropout_info=dropout_info,
                privacy_params=privacy_params
            )

            if member_name == "Early":
                truncate_idx = get_truncate_idx(
                    arr=privacy_spent_list,
                    threshold=target_epsilon
                )
                privacy_spent_list = privacy_spent_list[:truncate_idx]

            result["privacy"] = privacy_spent_list[-1]

    return result


def main(plot_plan_file):
    start_time = time.perf_counter()

    print("Staring to extract data...")
    with open(os.path.join(file_dir, plot_plan_file), 'r') as fin:
        plot_plan = yaml.load(fin, Loader=yaml.FullLoader)
    plots = plot_plan["plots"]

    raw_data_path = os.path.join(file_dir, plot_plan["raw_data"])
    # if True:
    if not os.path.isfile(raw_data_path):
        raw_data = {}
        for group in plot_plan["source"]:
            group_name = group["group"]
            member_data = group["member"]
            if member_data is None:
                continue

            if group_name not in raw_data:
                raw_data[group_name] = {}

            for exp_data in member_data:
                member_name = exp_data["value"]
                relative_folder = exp_data["folder"]
                exp_name = exp_data["exp_name"]

                print(f"\tExtracting data for {group_name} and {exp_name}...")
                _data = extract_data(
                    group_name=group_name,
                    member_name=member_name,
                    exp_name=exp_name,
                    parent_folder=os.path.join(file_dir, relative_folder),
                    plots=plots
                )
                raw_data[group_name][member_name] = _data
        with open(raw_data_path, 'wb') as fout:
            pickle.dump(raw_data, fout)
    else:
        with open(raw_data_path, 'rb') as fin:
            raw_data = pickle.load(fin)

    duration = round(time.perf_counter() - start_time, 4)
    print(f"Data extracted. {duration} secs are used. Starting to plot...")

    start_time = time.perf_counter()
    log_lines = []
    for plot_dict in plots:
        plot_type = plot_dict["type"]
        plot_params = plot_dict["params"]
        xlabel = plot_dict["xlabel"]
        ylabel = plot_dict["ylabel"]

        if plot_type == "dropout-acc-line":
            data_to_plot = []

            for group_name, group_dict in raw_data.items():
                x = []
                y = []
                for member_name, _data in group_dict.items():
                    acc_info = _data["acc_info"]

                    x.append(member_name)
                    # to percent
                    y.append(acc_info[used_acc_metric] * 100)
                    acc_info_to_print = {
                        "max": acc_info["max"] * 100,
                        "last-10-avg": round(acc_info["last-10-avg"] * 100, 2),
                        "last-20-avg": round(acc_info["last-20-avg"] * 100, 2),
                        "last-30-avg": round(acc_info["last-30-avg"] * 100, 2),
                    }
                    log_line = f"\t\t{group_name} {member_name}: {acc_info_to_print}"
                    print(log_line)
                    log_lines.append(log_line + '\n')

                data_to_plot.append({
                    'label': group_name,
                    'x': x,
                    'y': y
                })

            plot_path = os.path.join(file_dir, plot_dict["path"])
            complex_plot(
                data_to_plot,
                xlabel,
                ylabel,
                plot_path,
                params=plot_params,
                both_type=True,
                preprocess_acc=False
            )
        elif plot_type == "dropout-acc":  # group bar
            data_to_plot = {}
            for group_name, group_dict in raw_data.items():
                if group_name not in data_to_plot:
                    data_to_plot[group_name] = {}
                for member_name, member_dict in group_dict.items():
                    if member_name not in data_to_plot[group_name]:
                        data_to_plot[group_name][member_name] = {}
                    for element_name, element_dict in member_dict.items():
                        if "perplexity" in ylabel.lower():  # perplexity
                            data_to_plot[
                                group_name][member_name][element_name] \
                                    = int(element_dict[used_acc_metric])

                            acc_info_to_print = {
                                "min": element_dict["min"],
                                "last-10-avg": round(element_dict["last-10-avg"], 2),
                                "last-20-avg": round(element_dict["last-20-avg"], 2),
                                "last-30-avg": round(element_dict["last-30-avg"], 2),
                            }
                        else:  # accuracy
                            data_to_plot[
                                group_name][member_name][element_name] = \
                                element_dict[used_acc_metric] * 100
                            acc_info_to_print = {
                                "max": element_dict["max"] * 100,
                                "last-10-avg": round(element_dict["last-10-avg"] * 100, 2),
                                "last-20-avg": round(element_dict["last-20-avg"] * 100, 2),
                                "last-30-avg": round(element_dict["last-30-avg"] * 100, 2),
                            }
                        log_line = f"\t\t{group_name} {member_name}: {acc_info_to_print}"
                        print(log_line)
                        log_lines.append(log_line + '\n')

            stacked_bar_plot(
                data=data_to_plot,
                x_label=xlabel,
                y_label=ylabel,
                figure_file_name=os.path.join(file_dir, plot_dict["path"]),
                params=plot_dict["params"],
                both_type=True,
            )
        else:  # dropout-privacy
            intermediate_dict = {}
            for dropout_rate, d in raw_data.items():
                for method, data in d.items():
                    if method not in intermediate_dict:
                        intermediate_dict[method] = {}
                    intermediate_dict[method][dropout_rate] = data["privacy"]
            print(intermediate_dict)

            data_to_plot = []
            for label, label_dict in intermediate_dict.items():
                t = {
                    "label": label,
                    "x": [],
                    "y": []
                }
                for dropout_rate, privacy in label_dict.items():
                    t["x"].append(dropout_rate)
                    t["y"].append(privacy)
                data_to_plot.append(t)

            plot_path = os.path.join(file_dir, plot_dict["path"])
            complex_plot(
                data_to_plot,
                x_label=xlabel,
                y_label=ylabel,
                figure_file_name=plot_path,
                params=plot_params,
                both_type=True
            )

    log_path = os.path.join(file_dir, plot_dict["path"] + '.txt')
    with open(log_path, 'w') as fout:
        fout.writelines(log_lines)

    duration = round(time.perf_counter() - start_time, 4)
    print(f"Plots are ready. {duration} secs are used.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please specify the plot plan.")
    else:
        main(sys.argv[1])
