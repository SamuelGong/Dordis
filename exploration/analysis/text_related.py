import os
import json
import numpy as np
from backend import ROUND, COORDINATOR, OVERALL, AGGREGATION, \
    TIME_METRICS, NETWORK_METRICS, RAW_DATA, reject_outlier, \
    SEND_DATA_MB, RECV_DATA_MB, START_TIME, END_TIME

output_rel = "stat.json"


def basic_metrics(l, round_para):
    # clean_l = reject_outlier(np.array(l)).tolist()
    clean_l = l
    return {
        'l': [round(i, 2) for i in l],
        'sum': round(np.sum(clean_l), round_para),
        'mean': round(np.mean(clean_l), round_para),
        'median': round(np.median(clean_l), round_para),
        'std': round(np.std(clean_l), round_para)
    }


def analyze_round_time(result_dict):
    coordinator_dict = result_dict[COORDINATOR][ROUND]
    round_time_list = []
    for round_idx, round_dict in coordinator_dict.items():
        start_time = round_dict[OVERALL][START_TIME]
        end_time = round_dict[OVERALL][END_TIME]
        round_time_list.append(end_time - start_time)

    agg_time_list = []
    for round_idx, round_dict in coordinator_dict.items():
        start_time = round_dict[AGGREGATION][START_TIME]
        # print(round_dict)
        if END_TIME not in round_dict[AGGREGATION]:
            # because the program may break, for example,
            # at server_use_output stage because no sufficient
            # clients to sample
            break
        end_time = round_dict[AGGREGATION][END_TIME]
        agg_time_list.append(end_time - start_time)

    res = {
        "all": basic_metrics(round_time_list, round_para=3),
        "agg": basic_metrics(agg_time_list, round_para=3),
    }
    return res


def analyze_round_network(result_dict):
    coordinator_dict = result_dict[COORDINATOR][ROUND]
    send_data_list, recv_data_list, total_data_list = [], [], []
    agg_send_data_list, agg_recv_data_list, agg_total_data_list = [], [], []
    for round_idx, round_dict in coordinator_dict.items():
        send_data = round_dict[OVERALL][SEND_DATA_MB]
        recv_data = round_dict[OVERALL][RECV_DATA_MB]
        send_data_list.append(send_data)
        recv_data_list.append(recv_data)
        total_data_list.append(send_data + recv_data)

        agg_send_data = round_dict[AGGREGATION][SEND_DATA_MB]
        agg_recv_data = round_dict[AGGREGATION][RECV_DATA_MB]
        agg_send_data_list.append(agg_send_data)
        agg_recv_data_list.append(agg_recv_data)
        agg_total_data_list.append(agg_send_data + agg_recv_data)

    res = {
        "send": basic_metrics(send_data_list, round_para=6),
        "recv": basic_metrics(recv_data_list, round_para=6),
        "total": basic_metrics(total_data_list, round_para=6),
        "agg_send": basic_metrics(agg_send_data_list, round_para=6),
        "agg_recv": basic_metrics(agg_recv_data_list, round_para=6),
        "agg_total": basic_metrics(agg_total_data_list, round_para=6)
    }
    return res


def nice_json_dump(obj, output_path):
    with open(output_path, 'w+') as output_file:
        json.dump(
            obj=obj,
            fp=output_file,
            indent=4,
            separators=(',', ': ')
        )


def text_summary(task_folder, result_dict):
    round_time_metric = analyze_round_time(result_dict)
    round_network_metric = analyze_round_network(result_dict)

    output_path = os.path.join(task_folder, output_rel)
    nice_json_dump(
        obj={
            TIME_METRICS: round_time_metric,
            NETWORK_METRICS: round_network_metric,
            RAW_DATA: result_dict
        },
        output_path=output_path
    )
