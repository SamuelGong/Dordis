import os
import datetime

import numpy as np

COORDINATOR = "coordinator"
ROUND = "rounds"
CHUNK = "chunks"
PHASE = "phases"
OVERALL = "overall"
AGGREGATION = "aggregation"
START_TIME = "start_time"
END_TIME = "end_time"
SEND_DATA_MB = "send_data_mb"
RECV_DATA_MB = "recv_data_mb"
TIME_METRICS = "Round time metric (s)"
NETWORK_METRICS = "Round network metric (MB)"
RAW_DATA = "Raw data"

APP_METRICS = "App metrics"
TESTING_ACCURACY = "Testing accuracy"
MSE = "MSE"
mse_mode = "[Sum]"
# mse_mode = "[Mean]"


log_rel = "log.txt"
log_time_fmt = "(%Y-%m-%d) %H:%M:%S.%f"


def reject_outlier(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def extract_timestamp(line, fmt):
    timestamp_str = line.split('[')[2].split(']')[0]
    dt_obj = datetime.datetime.strptime(timestamp_str, fmt)
    return dt_obj


def get_rel_time(line, fmt, base):
    target_time = extract_timestamp(line, fmt)
    return (target_time - base).total_seconds()


def find_substr(line, preceding_string,
                succeeding_separator, mode="str"):
    temp_idx = line.find(preceding_string) + len(preceding_string)
    result_str = line[temp_idx:].split(succeeding_separator)[0]
    if mode == "int":
        return int(result_str)
    elif mode == "float":
        return float(result_str)
    else:
        return result_str


def add_stage_average_metrics(result_dict):
    temp_dict = {}

    for round_idx, round_dict \
            in result_dict[COORDINATOR][ROUND].items():
        first_chunk_dict = round_dict[CHUNK][0][PHASE]
        num_phases = len(first_chunk_dict.keys())

        # TODO: avoid hard-coding
        if num_phases == 12:
            for stage_idx in range(5):
                if stage_idx not in temp_dict:
                    temp_dict[stage_idx] = {
                        'duration': []
                    }

                duration = 0
                if stage_idx == 0:
                    for phase_idx in range(1, 5):  # TODO: avoid hard-coding
                        duration += result_dict['average_view'][phase_idx]['duration']
                elif stage_idx == 1:
                    for phase_idx in range(5, 6):
                        duration += result_dict['average_view'][phase_idx]['duration']
                elif stage_idx == 2:
                    for phase_idx in range(6, 9):
                        duration += result_dict['average_view'][phase_idx]['duration']
                elif stage_idx == 3:
                    for phase_idx in range(9, 10):
                        duration += result_dict['average_view'][phase_idx]['duration']
                else:
                    for phase_idx in range(10, 11):
                        duration += result_dict['average_view'][phase_idx]['duration']

                temp_dict[stage_idx]["duration"] \
                    .append(duration)
        elif num_phases == 8:
            for stage_idx in range(5):
                if stage_idx not in temp_dict:
                    temp_dict[stage_idx] = {
                        'duration': []
                    }

                duration = 0
                if stage_idx == 0:
                    for phase_idx in range(1, 2):  # TODO: avoid hard-coding
                        duration += result_dict['average_view'][phase_idx]['duration']
                elif stage_idx == 1:
                    for phase_idx in range(2, 3):
                        duration += result_dict['average_view'][phase_idx]['duration']
                elif stage_idx == 2:
                    for phase_idx in range(3, 5):
                        duration += result_dict['average_view'][phase_idx]['duration']
                elif stage_idx == 3:
                    for phase_idx in range(5, 6):
                        duration += result_dict['average_view'][phase_idx]['duration']
                else:
                    for phase_idx in range(6, 7):
                        duration += result_dict['average_view'][phase_idx]['duration']

                temp_dict[stage_idx]["duration"] \
                    .append(duration)

    # average over round
    dict_to_add = {}
    if temp_dict:
        for phase_idx, phase_dict in temp_dict.items():
            dict_to_add[phase_idx] = {
                "duration": round(np.average(phase_dict["duration"]), 2),
            }

        result_dict["average_stage_view"] = dict_to_add


def add_phase_average_metrics(result_dict):
    temp_dict = {}

    for round_idx, round_dict \
            in result_dict[COORDINATOR][ROUND].items():
        first_chunk_dict = round_dict[CHUNK][0][PHASE]
        chunks = round_dict[CHUNK].keys()
        num_chunks = len(chunks)
        phases = list(first_chunk_dict.keys())

        for phase_idx in phases:
            if phase_idx not in temp_dict:
                temp_dict[phase_idx] = {
                    'duration': [],
                    'traffic': []
                }

            duration = 0.0
            send_data_mb = 0.0
            recv_data_mb = 0.0
            for chunk_idx in chunks:
                chunk_dict = round_dict[CHUNK][chunk_idx][PHASE]
                start_time = chunk_dict[phase_idx]['start_time']
                # print(round_idx, chunk_idx, phase_idx, chunk_dict)
                end_time = chunk_dict[phase_idx]["end_time"]
                duration += end_time - start_time

                if 'send_data_mb' in first_chunk_dict[phase_idx]:
                    send_data_mb += chunk_dict[phase_idx]["send_data_mb"]
                if 'recv_data_mb' in first_chunk_dict[phase_idx]:
                    recv_data_mb += chunk_dict[phase_idx]["recv_data_mb"]
            network_mb = send_data_mb + recv_data_mb

            # TODO: avoid hard-coding
            if phase_idx == 0 or phase_idx == len(phases) - 1:
                duration /= num_chunks
                network_mb /= num_chunks

            temp_dict[phase_idx]["duration"]\
                .append(duration)
            temp_dict[phase_idx]["traffic"]\
                .append(network_mb)

    # average over round
    dict_to_add = {}
    for phase_idx, phase_dict in temp_dict.items():
        l = phase_dict["duration"]
        # l = (reject_outlier(np.array(l))).tolist()
        dict_to_add[phase_idx] = {
            "duration": round(np.average(l), 2),
            "duration_std": round(np.std(l), 2),
            "duration_list": [round(i, 2) for i in phase_dict["duration"]],
            "traffic": round(np.average(phase_dict["traffic"]), 2)
        }

    result_dict["average_view"] = dict_to_add


def extract_log(task_folder, phase_info, chunk_meta,
                app_info, mode="time"):
    path = None
    phases = sorted(list(phase_info.keys()))
    chunk_size_dict, data_dim, \
        padded_zeros_at_the_last_chunk = chunk_meta
    num_chunks = len(list(chunk_size_dict.keys()))
    # is_fl = app_info["is_fl"]
    test_metric = app_info["test_metric"]

    for item in os.listdir(task_folder):
        if COORDINATOR in item:
            path = os.path.join(task_folder, item)
            if os.path.isdir(path):
                break
    if path is None:
        raise ValueError(f"No folder in {task_folder} "
                         f"contains a folder "
                         f"with {COORDINATOR} in its name.")

    round_idx = -1
    result_dict = {}
    coordinator_start_time = None
    log_path = os.path.join(path, log_rel)
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()

        for line in lines:
            if "[INFO]" not in line:
                continue

            if "Starting a server" in line:
                coordinator_start_time = extract_timestamp(
                    line=line,
                    fmt=log_time_fmt
                )
                result_dict[COORDINATOR] = {
                    ROUND: {}
                }
            elif "Starting round" in line:
                round_idx = find_substr(
                    line=line,
                    preceding_string="round ",
                    succeeding_separator='.',
                    mode="int"
                )
                result_dict[COORDINATOR][ROUND][round_idx] = {
                    CHUNK: {}
                }

                if mode in ["time", "all"]:
                    time_rel = get_rel_time(
                        line=line,
                        fmt=log_time_fmt,
                        base=coordinator_start_time
                    )
                    result_dict[
                        COORDINATOR][ROUND][round_idx].update({
                        OVERALL: {
                            START_TIME: time_rel
                        }
                    })
            elif "Round" in line and "ended" in line:
                temp_round_idx = find_substr(
                    line=line,
                    preceding_string="Round ",
                    succeeding_separator=' ',
                    mode="int"
                )
                assert round_idx == temp_round_idx

                time_rel = get_rel_time(
                    line=line,
                    fmt=log_time_fmt,
                    base=coordinator_start_time
                )

                if mode in ["time", "all"]:
                    result_dict[
                        COORDINATOR][ROUND][round_idx][OVERALL]\
                        .update({
                        'end_time': time_rel
                    })

                    # quick fix for last round's missing Done
                    # TODO: futher debug this problem
                    last_phase = len(phases) - 1
                    for chunk_idx in range(num_chunks):
                        if last_phase in result_dict[
                            COORDINATOR][ROUND][round_idx][
                            CHUNK][chunk_idx][PHASE] \
                                and END_TIME not in result_dict[
                            COORDINATOR][ROUND][round_idx][
                            CHUNK][chunk_idx][PHASE][last_phase]:
                            # because the program may not necessarily break at the last phase
                            result_dict[
                                COORDINATOR][ROUND][round_idx][
                                CHUNK][chunk_idx][PHASE][last_phase][END_TIME] = time_rel

                # seems that only the first round is within expectation
                # do not do that to profiling-use data!
                # if is_fl and round_idx == 0:
                #     break
            elif "Phase" in line:
                time_rel = get_rel_time(
                    line=line,
                    fmt=log_time_fmt,
                    base=coordinator_start_time
                )
                round_idx = find_substr(
                    line=line,
                    preceding_string="[Round ",
                    succeeding_separator="]",
                    mode="int"
                )
                chunk_idx = find_substr(
                    line=line,
                    preceding_string="[Chunk ",
                    succeeding_separator="]",
                    mode="int"
                )
                phase_idx = find_substr(
                    line=line,
                    preceding_string="[Phase ",
                    succeeding_separator="]",
                    mode="int"
                )

                if mode in ["network", "all"] \
                        and "payload data" in line:
                    if "Sent" in line:
                        sent_size_in_mb = find_substr(
                            line=line,
                            preceding_string="Sent ",
                            succeeding_separator=" ",
                            mode="float"
                        )
                        if SEND_DATA_MB not in result_dict[
                            COORDINATOR][ROUND][round_idx][
                            CHUNK][chunk_idx][PHASE][phase_idx]:
                            result_dict[
                                COORDINATOR][ROUND][round_idx][
                                CHUNK][chunk_idx][PHASE][phase_idx].update({
                                SEND_DATA_MB: 0.0
                            })
                        result_dict[
                            COORDINATOR][ROUND][round_idx][
                            CHUNK][chunk_idx][PHASE][
                            phase_idx][SEND_DATA_MB] += sent_size_in_mb

                        if phases[1] <= phase_idx <= phases[-2]: # TODO: avoid hard-coding
                            if SEND_DATA_MB not in result_dict[
                                COORDINATOR][ROUND][round_idx][AGGREGATION]:
                                result_dict[COORDINATOR][ROUND][
                                    round_idx][AGGREGATION].update({
                                    SEND_DATA_MB: sent_size_in_mb
                                })
                            else:
                                original_received_size = result_dict[COORDINATOR][
                                    ROUND][round_idx][AGGREGATION][SEND_DATA_MB]
                                result_dict[COORDINATOR][ROUND][round_idx][AGGREGATION][
                                    SEND_DATA_MB] = original_received_size + sent_size_in_mb
                    elif "Received" in line:
                        received_size_in_mb = find_substr(
                            line=line,
                            preceding_string="Received ",
                            succeeding_separator=" ",
                            mode="float"
                        )
                        if RECV_DATA_MB not in result_dict[
                            COORDINATOR][ROUND][round_idx][
                            CHUNK][chunk_idx][PHASE][phase_idx]:
                            result_dict[
                                COORDINATOR][ROUND][round_idx][
                                CHUNK][chunk_idx][PHASE][phase_idx].update({
                                RECV_DATA_MB: 0.0
                            })
                        result_dict[
                            COORDINATOR][ROUND][round_idx][
                            CHUNK][chunk_idx][PHASE][
                            phase_idx][RECV_DATA_MB] += received_size_in_mb

                        if phases[1] <= phase_idx <= phases[-2]: # TODO: avoid hard-coding
                            if RECV_DATA_MB not in result_dict[
                                COORDINATOR][ROUND][round_idx][AGGREGATION]:
                                result_dict[COORDINATOR][ROUND][
                                    round_idx][AGGREGATION].update({
                                    RECV_DATA_MB: received_size_in_mb
                                })
                            else:
                                original_received_size = result_dict[COORDINATOR][
                                    ROUND][round_idx][AGGREGATION][RECV_DATA_MB]
                                result_dict[COORDINATOR][ROUND][round_idx][AGGREGATION][
                                    RECV_DATA_MB] = original_received_size + received_size_in_mb
                if "Phase started" in line:
                    if chunk_idx not in result_dict[
                            COORDINATOR][ROUND][round_idx][CHUNK]:
                        result_dict[COORDINATOR][
                            ROUND][round_idx][CHUNK][chunk_idx] = {
                            PHASE: {}
                        }

                    result_dict[
                        COORDINATOR][ROUND][round_idx][
                        CHUNK][chunk_idx][PHASE][phase_idx] = {
                        START_TIME: time_rel
                    }
                    if mode in ["time", "all"] \
                            and chunk_idx == 0 \
                            and phase_idx == phases[1]:  # TODO: avoid hard-coding
                        result_dict[
                            COORDINATOR][ROUND][round_idx] \
                            .update({
                            AGGREGATION: {
                                'start_time': time_rel
                            }})
                elif "Phase done" in line:
                    result_dict[
                        COORDINATOR][ROUND][round_idx][
                        CHUNK][chunk_idx][PHASE][phase_idx].update({
                            END_TIME: time_rel
                    })
                    if mode in ["time", "all"] \
                            and chunk_idx == num_chunks - 1 \
                            and phase_idx == phases[-2]:  # TODO: avoid hard-coding
                        result_dict[
                            COORDINATOR][ROUND][round_idx][AGGREGATION] \
                            .update({
                            'end_time': time_rel
                        })
                elif test_metric == "accuracy" and "Testing accuracy" in line:
                    testing_acc = find_substr(
                        line=line,
                        preceding_string="Testing accuracy: ",
                        succeeding_separator=".\n",
                        mode="float"
                    )
                    result_dict[
                        COORDINATOR][ROUND][round_idx].update({
                        APP_METRICS: {
                            TESTING_ACCURACY: {
                                'value': testing_acc,
                                'time': time_rel
                            }
                        }
                    })
                elif test_metric == "mse" and "MSE" in line and mse_mode in line:
                    _mse = find_substr(
                        line=line,
                        preceding_string="MSE: ",
                        succeeding_separator=".\n",
                        mode="float"
                    )
                    chunk_size = chunk_size_dict[chunk_idx]
                    if chunk_idx == num_chunks - 1 and padded_zeros_at_the_last_chunk > 0:
                        chunk_size -= padded_zeros_at_the_last_chunk
                    mse = _mse * chunk_size / data_dim  # by definition
                    if APP_METRICS not in result_dict[
                        COORDINATOR][ROUND][round_idx]:
                        result_dict[
                            COORDINATOR][ROUND][round_idx].update({
                            APP_METRICS: {
                                MSE: {
                                    'value': mse,
                                    'value_list': [_mse],
                                    'time': [time_rel]
                                }
                            }
                        })
                    else:
                        result_dict[COORDINATOR][ROUND][round_idx][
                            APP_METRICS][MSE]['value'] += mse
                        result_dict[COORDINATOR][ROUND][round_idx][
                            APP_METRICS][MSE]['value_list'].append(_mse)
                        result_dict[COORDINATOR][ROUND][round_idx][
                            APP_METRICS][MSE]['time'].append(time_rel)

    if mode in ["network", "all"]:
        result_dict = preprocess_network_data(result_dict)

    add_phase_average_metrics(result_dict)
    add_stage_average_metrics(result_dict)
    return result_dict, coordinator_start_time


def preprocess_network_data(result_dict):
    res = None
    round_para = 6

    if COORDINATOR in result_dict:
        coordinator_dict = result_dict[COORDINATOR]
        for round_idx, round_dict in coordinator_dict[ROUND].items():
            send_data_total_mb = 0.0
            recv_data_total_mb = 0.0
            for chunk_idx, chunk_dict in round_dict[CHUNK].items():
                for phase_idx, phase_dict in chunk_dict[PHASE].items():
                    if SEND_DATA_MB in phase_dict:
                        phase_dict[SEND_DATA_MB] = round(
                            phase_dict[SEND_DATA_MB], round_para)
                        send_data_total_mb += phase_dict[SEND_DATA_MB]
                    if RECV_DATA_MB in phase_dict:
                        phase_dict[RECV_DATA_MB] = round(
                            phase_dict[RECV_DATA_MB], round_para)
                        recv_data_total_mb += phase_dict[RECV_DATA_MB]
            if OVERALL not in round_dict:
                round_dict[OVERALL] = {}
            round_dict[OVERALL].update({
                SEND_DATA_MB: round(send_data_total_mb, round_para),
                RECV_DATA_MB: round(recv_data_total_mb, round_para)
            })

            if AGGREGATION in round_dict:
                if SEND_DATA_MB in round_dict[AGGREGATION]:
                    round_dict[AGGREGATION][SEND_DATA_MB] = round(
                        round_dict[AGGREGATION][SEND_DATA_MB], round_para
                    )
                if RECV_DATA_MB in round_dict[AGGREGATION]:
                    round_dict[AGGREGATION][RECV_DATA_MB] = round(
                        round_dict[AGGREGATION][RECV_DATA_MB], round_para
                    )
        res = result_dict

    return res
