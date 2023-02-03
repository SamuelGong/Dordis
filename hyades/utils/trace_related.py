import numpy as np
import pickle
import os
import logging

file_dir = os.path.dirname(os.path.realpath(__file__))
trace_file = os.path.join(file_dir, "client_behave_trace_old.pkl")


def offseting_trace_data(trace_data, offset):
    result = {}
    client_ids = sorted(list(trace_data.keys()))
    for client_id in client_ids:
        client_trace = trace_data[client_id]
        trace_active_list = np.array(client_trace["active"])
        trace_inactive_list = np.array(client_trace["inactive"])

        trace_active_list -= offset
        trace_inactive_list -= offset

        client_trace["active"] = trace_active_list.tolist()
        client_trace["inactive"] = trace_inactive_list.tolist()
        result[client_id] = client_trace

    return result


def read_trace(total_duration, offset):
    with open(trace_file, "rb") as fin:
        trace_data = pickle.load(fin)

    # selecting different interval to analyse
    # according to offset
    trace_data = offseting_trace_data(trace_data, offset)

    LABEL = 0
    raw_data = {}
    client_ids = sorted(list(trace_data.keys()))
    finish_time_dict = {}

    for client_id in client_ids:
        client_trace = trace_data[client_id]
        trace_active_list = client_trace["active"]
        trace_inactive_list = client_trace["inactive"]
        finish_time_dict[client_id] = client_trace["finish_time"]

        active_list = []
        exit_loop = False
        for idx, active_time in enumerate(trace_active_list):

            if active_time >= total_duration:
                exit_loop = True
            elif active_time >= 0:
                inactive_time = trace_inactive_list[idx]
                if inactive_time > total_duration:
                    end = total_duration
                else:
                    end = inactive_time

                active_list.append([LABEL, active_time, end])
            else:  # 0 < active_time < total_duration
                inactive_time = trace_inactive_list[idx]
                if not inactive_time <= 0:
                    if inactive_time > total_duration:
                        end = total_duration
                    else:
                        end = inactive_time

                    active_list.append([LABEL, 0, end])

            if exit_loop:
                break

        if len(active_list) > 0:
            raw_data[client_id] = active_list

    keys = list(raw_data.keys())
    np.random.shuffle(keys)
    raw_data_shuffled = {
        k:raw_data[k] for k in keys
    }
    return raw_data_shuffled, finish_time_dict


def sort_after_selection_dropout(raw_data, num_rounds,
                                 round_duration, aggregation_latency):
    sorted_reference_dropout_times = {}
    for client_id, active_list in raw_data.items():
        max_selection_times = 0
        corresponding_dropout_times = 0

        for round_id in range(num_rounds):
            selection_time = round_id * round_duration
            for tu in active_list:
                _, start_time, end_time = tu
                if start_time <= selection_time < end_time:
                    max_selection_times += 1

                    aggregation_time = selection_time + aggregation_latency
                    if aggregation_time > end_time:
                        corresponding_dropout_times += 1
                    break

        if max_selection_times == 0:
            continue

        sorted_reference_dropout_times[client_id] = corresponding_dropout_times

    sorted_reference_dropout_times = {k: v for k, v in
                                      sorted(sorted_reference_dropout_times.items(),
                                             key=lambda item: item[1])}
    # print(sorted_reference_dropout_times)
    client_ids_sorted = list(sorted_reference_dropout_times.keys())
    raw_data_sorted = {
        k: raw_data[k] for k in client_ids_sorted
    }
    return raw_data_sorted


def raw_data_subsampling(raw_data_sorted, trace_subsampling_rank,
                         total_clients):
    client_ids_sorted = list(raw_data_sorted.keys())
    trace_pool_size = len(client_ids_sorted)

    num_subsets = trace_pool_size // total_clients
    which_to_pick = int(np.floor((num_subsets - 1) * trace_subsampling_rank / 100))

    print(f"\tTrace pool size: {trace_pool_size}, "
          f"# disjoint_trace_subset: {num_subsets}, "
          f"Taking the {which_to_pick}-th subset (rank top{trace_subsampling_rank}%).")

    client_ids_subsampled = client_ids_sorted[
                            which_to_pick * total_clients:(which_to_pick + 1) * total_clients]
    np.random.shuffle(client_ids_subsampled)
    raw_data_subsampled = {
        k: raw_data_sorted[k] for k in client_ids_subsampled
    }
    return raw_data_subsampled


def find_available(raw_data, current_time):
    available_clients = []
    related_period_dict = {}
    for client_id, active_list in raw_data.items():
        for tu in active_list:
            _, start_time, end_time = tu
            if start_time <= current_time < end_time:
                available_clients.append(client_id)
                related_period_dict[client_id] = (start_time, end_time)
                break

    # if len(available_clients) == 2:
    #     logging.info(f"[Debug] {current_time} {raw_data}.")
    return available_clients, related_period_dict


def find_surviving(related_period_dict, current_time):
    res = []
    for client_id, related_period in related_period_dict.items():
        start_time, end_time = related_period
        if start_time <= current_time < end_time:
            res.append(client_id)

    return res
