import os


def print_a_metric_dim_3(formatted_data, metric, params=None):
    percent_dict = {}
    if params is not None and "percent" in params:
        percent_dict = params["percent"]

    lines = [metric.upper() + ":\n"]
    for row_label, row_data in formatted_data.items():
        text_list = []
        for column_label, cell_list in row_data.items():
            data = [d[metric] for d in cell_list]

            data_text = []
            for idx, item in enumerate(data):
                item_str = str(item)
                if metric == "traffic":

                    if item >= 1024:
                        item_str = str(round(item / 1024, 2))
                        item_str += "G"
                    else:
                        item_str += "M"
                if percent_dict and idx in percent_dict:
                    baseline = data[percent_dict[idx]]
                    quotient = item / baseline
                    quotient = abs(1 - quotient)
                    quotient *= 100
                    quotient = round(quotient)
                    if quotient == 0:
                        data_text.append(item_str)
                    else:
                        data_text.append(item_str + f"({quotient}%)")
                else:
                    data_text.append(item_str)

            text = "/ ".join(data_text)
            text_list.append(text)
        line = " | ".join(text_list)
        lines.append(line + "\n")

    lines.append("\n")
    print(lines)
    return lines


def table_dim_3(parent_folder, data, table_rel, params=None):
    exp_types = list(data.keys())
    num_exp_types = len(exp_types)
    exp_types_dict = {k: idx for idx, k in enumerate(exp_types)}
    metrics = []

    formatted_data = {}
    for exp_type, d in data.items():
        for row_label, row_dict in d.items():
            for col_label, col_dict in row_dict.items():
                if row_label not in formatted_data:
                    formatted_data[row_label] = {}
                if col_label not in formatted_data[row_label]:
                    formatted_data[row_label][col_label] \
                        = [None] * num_exp_types

                formatted_data[
                    row_label][col_label][
                    exp_types_dict[exp_type]] = col_dict
                for k in col_dict.keys():
                    if k not in metrics:
                        metrics.append(k)

    lines = []
    for metric in metrics:
        lines += print_a_metric_dim_3(formatted_data, metric, params)

    table_path = os.path.join(parent_folder, table_rel)
    with open(table_path, 'w') as fout:
        fout.writelines(lines)
