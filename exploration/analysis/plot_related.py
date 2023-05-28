import os
import numpy as np
import matplotlib.pyplot as plt
from backend import ROUND, COORDINATOR, CHUNK, PHASE
import matplotlib as mpl

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
output_rel = "time_seq"


qualitative_colors = [
    '#8dd3c7','#ffffb3','#bebada','#fb8072',
    '#80b1d3','#fdb462','#b3de69','#fccde5',
    '#d9d9d9','#bc80bd','#ccebc5'
]
sequantial_colors = [
    ['#fdb863', '#e66101'],
    ['#b2abd2', '#5e3c99']
]

bar_patterns = ["/" , "\\" , "-", "|" , "+" , "x", "o", "O", ".", "*"]
line_patterns = ["-", "--", ":", "-."]
dot_patterns = ["o", "v", "^"]
color_scheme_dict = {
    1: ['#f1a340'],
    2: ['#f1a340','#998ec3'],
    3: ['#e66101','#fdb863','#b2abd2'],
    4: ['#e66101','#fdb863','#b2abd2','#5e3c99'],
    5: ['#b35806','#f1a340','#fee0b6','#998ec3','#542788'],
    6: ['#b35806','#f1a340','#fee0b6','#d8daeb','#998ec3','#542788'],
    7: ['#b35806','#e08214','#fdb863','#fee0b6','#b2abd2','#8073ac','#542788'],
    8: ['#b35806','#e08214','#fdb863','#fee0b6','#d8daeb','#b2abd2','#8073ac','#542788']
}


def stacked_bar_plot(x_label, y_label, data, figure_file_name,
             params=None, both_type=False):
    customized_params = None
    if params and "customized" in params:
        customized_params = params["customized"]

    if params and "figsize" in params:
        _figsize = params["figsize"]
    else:
        _figsize = (2, 1.6)
    fig = plt.figure(figsize=_figsize, dpi=1200)
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if params and "fontsize" in params:
        fontsize = params["fontsize"]
    else:
        fontsize = 9

    if params and "bar_width" in params:
        bar_width = params["bar_width"]
    else:
        bar_width = 0.3

    # data preprocessing
    max_bar_height = -1
    bar_height_dict = {}  # for the use of percentage_first_element
    for group, group_dict in data.items():
        bar_height_dict[group] = {}
        for member, member_dict in group_dict.items():
            bottom = 0
            for element, value in member_dict.items():
                bottom += value
            bar_height_dict[group][member] = bottom
            if bottom > max_bar_height:
                max_bar_height = bottom

    if customized_params and 'yaxis_time' in customized_params \
            and customized_params['yaxis_time']:
        time_division_factor = 1.0
        if 60 <= max_bar_height < 7200:
            time_division_factor = 60
            y_label += ' (min)'
        elif max_bar_height >= 7200:
            time_division_factor = 3600
            y_label += ' (h)'
        else:
            y_label += ' (s)'

        for group, group_dict in data.items():
            for member, member_dict in group_dict.items():
                for element, value in member_dict.items():
                    member_dict[element] = value / time_division_factor
                bar_height_dict[group][member] /= time_division_factor
        max_bar_height /= time_division_factor

    first_group = data[list(data.keys())[0]]
    last_group = data[list(data.keys())[-1]]
    # num_members = len(list(first_group.keys()))
    num_members = len(list(last_group.keys()))
    first_member_of_first_group = first_group[list(first_group.keys())[0]]
    # num_elements = len(list(first_member_of_first_group.keys()))
    first_member_of_last_group = first_group[list(last_group.keys())[0]]
    num_elements = len(list(first_member_of_last_group.keys()))
    group_cnt = 0
    text_hoffset_factor = 1.0  # TODO: avoid hard-coding
    if "text_hoffset_factor" in customized_params:
        text_hoffset_factor = customized_params["text_hoffset_factor"]
    text_voffset_factor = 0.02
    group_bar_gap = 0.1
    if "group_bar_gap" in customized_params:
        group_bar_gap = customized_params["group_bar_gap"]
    hatch_density = 2
    if "hatch_density" in customized_params:
        hatch_density = customized_params["hatch_density"]
    mpl.rcParams['hatch.linewidth'] = 0.3
    percentage_first_element = False
    if customized_params \
            and "percentage_first_element" in customized_params:
        percentage_first_element = customized_params["percentage_first_element"]

    for group, group_dict in data.items():
        member_cnt = 0
        group_center = group_cnt
        group_leftmost_member_center \
            = group_center - (num_members - 1) \
              * (bar_width + group_bar_gap) / 2
        for member, member_dict in group_dict.items():
            bottom = 0
            element_cnt = 0
            member_center = group_leftmost_member_center \
                            + member_cnt * (bar_width + group_bar_gap)
            for element, value in member_dict.items():
                # so that legend can be plotted gracefully
                if "reverse_label" in customized_params:
                    first = element
                    second = member
                else:
                    first = member
                    second = element
                if "label_separator" in customized_params:
                    label_separator = customized_params["label_separator"]
                else:
                    label_separator = "-"

                # if group_cnt == 0:
                if group_cnt == len(data) - 1:  # last group
                    if num_elements == 1:  # no stacking
                        label = f"{first}"
                    else:
                        label = f"{first}{label_separator}{second}"
                else:
                    label = None

                if num_elements == 1:  # no stacking
                    color = color_scheme_dict[num_members][member_cnt]
                else:
                    color = sequantial_colors[member_cnt][element_cnt]
                ax.bar(
                    member_center, value,
                    width=bar_width, bottom=bottom,
                    label=label, color=color,
                    hatch=bar_patterns[(element_cnt + member_cnt * num_elements ) % len(bar_patterns)] * hatch_density,
                )
                bottom += value

                element_cnt += 1
                if element_cnt == num_elements:
                    text_fontsize = fontsize
                    if "text_fontsize" in customized_params:
                        text_fontsize = customized_params["text_fontsize"]
                    _text_hoffset_factor = text_hoffset_factor

                    if bottom < 10:  # so that small numbers will be "too left"
                        _text_hoffset_factor *= 0.7
                    ax.text(
                        member_center - bar_width * _text_hoffset_factor,
                        bottom + max_bar_height * text_voffset_factor,
                        str(round(bottom, 2)), fontsize=text_fontsize
                    )
                elif element_cnt == 1 \
                        and num_elements > 1 and percentage_first_element:
                    text_fontsize = fontsize
                    args = customized_params["percentage_first_element"]
                    if "text_fontsize" in args:
                        text_fontsize = args["text_fontsize"]
                    _text_hoffset_factor = text_hoffset_factor

                    _text_hoffset_factor_2 = 0.4
                    if "text_hoffset_factor_2" in args:
                        _text_hoffset_factor_2 = args["text_hoffset_factor_2"]

                    round_para = 0
                    if "round_para" in args:
                        round_para = args["round_para"]

                    overall_bar_height = bar_height_dict[group][member]
                    percentage = int(round(bottom / overall_bar_height * 100, round_para))
                    percentage_text = f"{percentage}%"
                    ax.text(
                        member_center - bar_width
                        * _text_hoffset_factor_2 * _text_hoffset_factor,
                        bottom / 2,
                        percentage_text, fontsize=text_fontsize,
                        weight='bold'
                    )
            member_cnt += 1
        group_cnt += 1

    xticks = range(len(data))
    ax.set_xticks(xticks)
    xtickslabels = list(data)
    ax.set_xticklabels(xtickslabels, fontsize=fontsize)

    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize)

    display_legend = True
    if 'display_legend' in params:
        display_legend = params['display_legend']
    if display_legend:
        legend_params = {
            'loc': 'best',
            'frameon': False,
            'fontsize': fontsize
        }
        if 'legend' in params:
            legend_params.update(params['legend'])

        legend_separate = customized_params["legend_separate"] \
            if (customized_params and "legend_separate" in customized_params) \
            else False

    if display_legend and not legend_separate:
        ax.legend(**legend_params)

    fig.savefig(
        figure_file_name,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if both_type:
        fig.savefig(
            figure_file_name + '.pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )

    if display_legend and legend_separate:
        legend_figure_params = {
            'figsize': (0.1, 0.1),
            'dpi': 1200
        }
        fig = plt.figure(**legend_figure_params)
        label_params = ax.get_legend_handles_labels()
        axl = fig.add_subplot(111, label='a')
        axl.axis(False)
        axl.legend(*label_params, **legend_params)
        fig.savefig(
            figure_file_name + '_legend',
            bbox_inches='tight',
            pad_inches=0.0
        )
        if both_type:
            fig.savefig(
                figure_file_name + '_legend.pdf',
                bbox_inches='tight',
                pad_inches=0.0
            )


def bar_plot(x_label, y_label, data, figure_file_name,
             params=None, both_type=False):
    customized_params = None
    if params and "customized" in params:
        customized_params = params["customized"]

    if params and "figsize" in params:
        _figsize = params["figsize"]
    else:
        _figsize = (2, 1.6)
    fig = plt.figure(figsize=_figsize, dpi=1200)
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    label_list = []
    value_list = []
    std_list = []
    for label, label_data in data.items():
        label_list.append(label)
        value_list.append(label_data["value"])
        if "std" in label_data:
            std_list.append(label_data["std"])
    max_value = max(value_list)

    if std_list:
        assert len(std_list) == len(value_list)

    if params and "fontsize" in params:
        _fontsize = params["fontsize"]
    else:
        _fontsize = 9

    if customized_params \
            and "xticks" in customized_params \
            and customized_params["xticks"] == "label":
        xticks = label_list
    else:
        xticks = np.arange(len(label_list))

    bar_width = 0.8
    if 'bar_width' in params:
        bar_width = params['bar_width']

    color_scheme = color_scheme_dict[len(value_list)]
    hatch_density = 2
    hatches = [bar_patterns[idx % len(bar_patterns)] * hatch_density
               for idx in range(len(value_list))]

    # ax.text
    round_params = 1
    if customized_params and "round_params" in customized_params:
        round_params = customized_params["round_params"]
    text_vertical_offset = max_value * 0.02
    text_horizontal_offset = round_params * 0.15

    if std_list:
        ax.bar(xticks, value_list, width=bar_width, color=color_scheme,
               ecolor='black', yerr=std_list, capsize=3,
               error_kw=dict(lw=1, capsize=3, capthick=0.5),
               hatch=hatches)
        for _idx, value in enumerate(value_list):
            xtick = xticks[_idx]
            std = std_list[_idx]
            ax.text(xtick - bar_width * text_horizontal_offset,
                    value + std + text_vertical_offset,
                    str(round(value, round_params)), fontsize=_fontsize)
    else:
        for _idx, value in enumerate(value_list):
            xtick = xticks[_idx]
            ax.bar(xtick, value, width=bar_width, color=color_scheme[_idx],
                   hatch=hatches[_idx])
            ax.text(xtick - bar_width * text_horizontal_offset,
                    value + text_vertical_offset, str(round(value, round_params)),
                    fontsize=_fontsize)
    plt.yticks(fontsize=_fontsize)

    if customized_params \
            and "xticks" in customized_params \
            and customized_params["xticks"] == "label":
        pass
    else:
        ax.set_xticks(xticks)
        if customized_params and "xticklabel_rotation" in customized_params:
            _rotation = customized_params["xticklabel_rotation"]
        else:
            _rotation = 0
        ax.set_xticklabels(label_list, fontsize=_fontsize,
                           rotation=_rotation)

    if y_label:
        ax.set_ylabel(y_label, fontsize=_fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=_fontsize)

    if customized_params and "ticklabel_sciformat" in customized_params:
        if 'x' in customized_params["ticklabel_sciformat"] \
                and customized_params["ticklabel_sciformat"]['x']:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if 'y' in customized_params["ticklabel_sciformat"] \
                and customized_params["ticklabel_sciformat"]['y']:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    fig.savefig(
        figure_file_name,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if both_type:
        fig.savefig(
            figure_file_name + '.pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )


def plot(x_label, x_value, y_label, y_value, figure_file_name,
         scatter=False, label_list=None):
    """Plot a figure."""
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    if not isinstance(y_value[0], list):
        y_value = [y_value]

    for idx, _y_value in enumerate(y_value):
        if label_list is not None:
            label = label_list[idx]
        else:
            label = None
        if not scatter:
            ax.plot(x_value, _y_value, label=label)
        else:
            ax.plot(x_value, _y_value, 'o', label=label)

    # ax.set_xlim(left=0)
    # ax.set_ylim((50, 100))
    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(x_label, fontsize=_fontsize)
    ax.legend(loc='best', frameon=False, fontsize=_fontsize)
    fig.savefig(figure_file_name, bbox_inches='tight', pad_inches=0.0)


def preprocess_accuracy(data, log=False, y_to_min_max=False):
    min_max = 1e32
    min_label = None

    # the first pass
    for _idx, _data in enumerate(data):
        y = _data['y']
        if 'label' in _data:
            label = _data['label']
        else:
            label = None

        m = max(y)
        if m < min_max:
            min_max = m
            min_label = label

    if log:
        print(f"min_label: {min_label}, min_max: {min_max}.")

    for _idx, _data in enumerate(data):
        label = _data['label']
        y = _data['y']

        _i = np.where(_data['y'] < min_max)[0]
        for j, k in enumerate(_i):
            if j > 0 and not k == _i[j-1] + 1:
                break
        _i = _i[:j]
        _i = np.append(_i, [_i[-1] + 1], axis=0)

        if log:
            last_avg_period = 10  # TODO: avoid hard-coding
            print(f'\t{label}: max_y: {round(max(_data["y"]), 2)}, '
                  f'last_y: {round(y[-1], 2)}, '
                  f'last_{last_avg_period}_avg: '
                  f'{round(np.mean(_data["y"][-last_avg_period:]), 2)}, '
                  f'x_to_min_max_y: {_i[-1]}')

        if y_to_min_max:
            _data['x'] = _data['x'][_i]
            _data['y'] = _data['y'][_i]


def smooth(x, y, y_std, smooth_args):
    type = smooth_args["type"]
    if type == "interval":
        interval = smooth_args["args"][0]
        new_x = np.array(x)[np.arange(0, len(x), interval)]
        new_y = np.array(y)[np.arange(0, len(y), interval)]
        if y_std is not None:
            new_y_std = np.array(y_std)[np.arange(0, len(y), interval)]
        else:
            new_y_std = None
    else:
        raise NotImplementedError

    return new_x, new_y, new_y_std


def complex_plot(data, x_label, y_label, figure_file_name,
                 params=None, both_type=False, preprocess_acc=False):

    customized_params = None
    if params and "customized" in params:
        customized_params = params["customized"]

    if params and "fontsize" in params:
        _fontsize = params["fontsize"]
    else:
        _fontsize = 9

    if params and "figsize" in params:
        _figsize = params["figsize"]
    else:
        _figsize = (2, 1.6)

    fig = plt.figure(figsize=_figsize, dpi=1200)
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    y_to_min_max = False
    if customized_params \
            and "y_to_min_max" in customized_params \
            and customized_params["y_to_min_max"]:
        y_to_min_max = True

    if preprocess_acc:
        preprocess_accuracy(data, log=True,
                            y_to_min_max=y_to_min_max)

    tmp_color_scheme = color_scheme_dict[len(data)]
    for _idx, _data in enumerate(data):
        if 'x' in _data:
            x = _data['x']
        else:
            x = _idx
        y = _data['y']
        if 'y_std' in _data and _data["y_std"]:
            y_std = _data["y_std"]
        else:
            y_std = None

        if customized_params and 'smooth' in customized_params:
            smooth_args = customized_params["smooth"]
            x, y, y_std = smooth(x, y, y_std, smooth_args)

        if 'label' in _data:
            label = _data['label']
        else:
            label = None
        if customized_params \
                and 'color_scheme' in customized_params \
                and customized_params['color_scheme']:
            color = tmp_color_scheme[_idx % len(tmp_color_scheme)]
        elif 'color' in _data:
            color = _data["color"]
        else:
            color = None

        if 'scatter' in _data and _data['scatter']:
            # ax.plot(x, y, 'o', label=label, color=color)
            if customized_params \
                    and "scatter_marker_size" in customized_params:
                s = customized_params["scatter_marker_size"]
                ax.scatter(x, y, label=label, color=color, s=s)
            else:
                ax.scatter(x, y, label=label, color=color)
        elif 'bar' in _data and _data['bar']:
            ax.bar(x, y, label=label, color=color)
        else:
            p = '-'
            if customized_params \
                    and "line_pattern" in customized_params \
                    and customized_params["line_pattern"]:
                p = line_patterns[_idx % len(line_patterns)]
            if customized_params \
                    and "dot_pattern" in customized_params \
                    and customized_params["dot_pattern"]:
                p = p + dot_patterns[_idx % len(dot_patterns)]

            ax.plot(x, y, p, label=label, color=color)

            if y_std:
                y = np.array(y)
                y_std = np.array(y_std)
                ax.fill_between(x, y - y_std, y + y_std, linewidth=0.0,
                                color=color, alpha=0.2)

        if customized_params \
                and "add_text_xticks" in customized_params:
            tick_d = customized_params["add_text_xticks"]
            tick_l = list(tick_d)

            xtick_cnt = 0
            for xi, yi in zip(x, y):
                if xtick_cnt in tick_l:
                    text_hoffset_factor \
                        = tick_d[xtick_cnt]["text_hoffset_factor"][_idx]
                    text_voffset_factor \
                        = tick_d[xtick_cnt]["text_voffset_factor"][_idx]
                    ax.text(xi - text_hoffset_factor,
                            yi + text_voffset_factor,
                            str(round(yi, 1)))
                xtick_cnt += 1

    # ax.set_xlim(left=0)
    if params and "ylim" in params:
        _ylim = params["ylim"]
        if len(_ylim) == 2:
            ax.set_ylim(_ylim)
        elif len(_ylim) == 1:
            ylim_right = ax.get_ylim()[1]
            ax.set_ylim(_ylim + [ylim_right])

    if params and "yscale" in params:
        ax.set_yscale(params['yscale'])

    if params and "xlim" in params:
        _xlim = params["xlim"]
        if len(_xlim) == 2:
            ax.set_xlim(_xlim)
        elif len(_xlim) == 1:
            xlim_right = ax.get_xlim()[1]
            ax.set_xlim(_xlim + [xlim_right])

    if params and "xticks" in params:
        ax.set_xticks(params['xticks'])
    if params and "yticks" in params:
        ax.set_yticks(params['yticks'])

    if customized_params:
        if "num_xticks" in customized_params:
            _num_xticks = customized_params["num_xticks"]
            current_xrlim = int(ax.get_xlim()[1])
            partition_size = current_xrlim // _num_xticks
            # print(current_xrlim, partition_size, _num_xticks)
            new_xticks = np.arange(0, current_xrlim, partition_size)
            ax.set_xticks(new_xticks)

    if y_label:
        ax.set_ylabel(y_label, fontsize=_fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=_fontsize)

    if customized_params and "ticklabel_sciformat" in customized_params:
        if 'x' in customized_params["ticklabel_sciformat"] \
                and customized_params["ticklabel_sciformat"]['x']:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if 'y' in customized_params["ticklabel_sciformat"] \
                and customized_params["ticklabel_sciformat"]['y']:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    if "grid" in params:
        grid_args = params["grid"]
        plt.grid(**grid_args)

    display_legend = True
    if 'display_legend' in params:
        display_legend = params['display_legend']
    if display_legend:
        legend_params = {
            'loc': 'best',
            'frameon': False,
            'fontsize': _fontsize
        }
        if 'legend' in params:
            legend_params.update(params['legend'])

        legend_separate = customized_params["legend_separate"] \
            if (customized_params and "legend_separate" in customized_params) \
            else False

    if display_legend and not legend_separate:
        ax.legend(**legend_params)

    fig.savefig(
        figure_file_name,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if both_type:
        fig.savefig(
            figure_file_name + '.pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )


    if display_legend and legend_separate:
        legend_figure_params = {
            'figsize': (0.1, 0.1),
            'dpi': 1200
        }
        fig = plt.figure(**legend_figure_params)
        label_params = ax.get_legend_handles_labels()
        axl = fig.add_subplot(111, label='a')
        axl.axis(False)
        axl.legend(*label_params, **legend_params)
        fig.savefig(
            figure_file_name + '_legend',
            bbox_inches='tight',
            pad_inches=0.0
        )
        if both_type:
            fig.savefig(
                figure_file_name + '_legend.pdf',
                bbox_inches='tight',
                pad_inches=0.0
            )


def multi_lat_cdf(label_dict, xlabel, ylabel, filename,
                  display_legend=True, both_type=False, params=None):
    customized_params = None
    if params and "customized" in params:
        customized_params = params["customized"]

    if params and "fontsize" in params:
        fontsize = params["fontsize"]
    else:
        fontsize = 9

    if params and "figsize" in params:
        figsize = params["figsize"]
    else:
        figsize = (1.2, 0.96)
    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    color_scheme = color_scheme_dict[len(label_dict)]
    idx = 0
    for label, d in label_dict.items():
        data = d['data']
        linestyle = d['linestyle']
        if 'color' in d:
            color = d['color']
        else:
            color = color_scheme[idx]
        data_sorted = sorted(data)
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        if "%" in ylabel:
            p *= 100
        ax.plot(data_sorted, p, color=color,
                label=label, linestyle=linestyle)
        idx += 1
    if 'xlim' in params:
        ax.set_xlim(params['xlim'])
    if 'ylim' in params:
        ax.set_ylim(params['ylim'])
    if 'yticks' in params:
        ax.set_yticks(params['yticks'])

    ax.minorticks_on()
    if "grid" in params:
        grid_args = params["grid"]
        plt.grid(**grid_args)

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.grid(True, which="both")
    if display_legend:
        legend_params = {
            'loc': 'best',
            'frameon': False,
            'fontsize': fontsize
        }
        if 'legend' in params:
            legend_params.update(params['legend'])

        legend_separate = customized_params["legend_separate"] \
            if (customized_params and "legend_separate" in customized_params) \
            else False

    if display_legend and not legend_separate:
        ax.legend(**legend_params)

    fig.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if both_type:
        fig.savefig(
            filename + '.pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )

    if display_legend and legend_separate:
        legend_figure_params = {
            'figsize': (0.1, 0.1),
            'dpi': 1200
        }
        fig = plt.figure(**legend_figure_params)
        label_params = ax.get_legend_handles_labels()
        axl = fig.add_subplot(111, label='a')
        axl.axis(False)
        axl.legend(*label_params, **legend_params)
        fig.savefig(
            filename + '_legend',
            bbox_inches='tight',
            pad_inches=0.0
        )
        if both_type:
            fig.savefig(
                filename + '_legend.pdf',
                bbox_inches='tight',
                pad_inches=0.0
            )


def plot_twinx_bar(data, x_label, y1_label, y2_label, y1_legend, y2_legend,
                   fig_save_path, display_legend=True,
                   both_type=False, params=None):
    customized_params = None
    if params and "customized" in params:
        customized_params = params["customized"]

    if params and "fontsize" in params:
        fontsize = params["fontsize"]
    else:
        fontsize = 9

    if params and "figsize" in params:
        figsize = params["figsize"]
    else:
        figsize = (1.2, 0.96)

    if params and "inner_bar_gap" in params:
        inner_bar_gap = params["inner_bar_gap"]
    else:
        inner_bar_gap = 0.1

    if params and "bar_width" in params:
        bar_width = params["bar_width"]
    else:
        bar_width = 0.3

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.xticks(fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)

    text_hoffset_factor = 1.0  # TODO: avoid hard-coding
    text_hoffset_factor_2 = 0.8
    if "text_hoffset_factor" in customized_params:
        text_hoffset_factor = customized_params["text_hoffset_factor"]
    text_voffset_factor = 0.05
    text_fontsize = fontsize
    if "text_fontsize" in customized_params:
        text_fontsize = customized_params["text_fontsize"]

    hatch_density = 2  # TODO: avoid hard-coding
    hatches = ["/" * hatch_density, "\\" * hatch_density]

    colors = ['#e66101', '#fdb863']
    xtick = 0
    max_y1 = -1
    max_y2 = -1
    for method, raw in data.items():
        if max_y1 < raw["y1"]:
            max_y1 = raw["y1"]
        if max_y2 < raw["y2"]:
            max_y2 = raw["y2"]

    for method, raw in data.items():
        x1 = xtick - inner_bar_gap / 2 - bar_width / 2
        y1 = raw["y1"]
        x2 = xtick + inner_bar_gap / 2 + bar_width / 2
        y2 = raw["y2"]

        if xtick == 0:
            _y1_label = y1_legend
            _y2_label = y2_legend
        else:
            _y1_label = None
            _y2_label = None

        ax.bar(x1, y1, bar_width, color=colors[0],
               label=_y1_label, hatch=hatches[0])
        ax2.bar(x2, y2, bar_width, color=colors[1],
                label=_y2_label, hatch=hatches[1])

        ax.text(
            x1 - bar_width * text_hoffset_factor,
            y1 + max_y1 * text_voffset_factor,
            str(round(y1, 1)), fontsize=text_fontsize
        )
        ax2.text(
            x2 - bar_width * text_hoffset_factor_2,
            y2 + max_y2 * text_voffset_factor,
            str(round(y2, 1)), fontsize=text_fontsize
        )
        xtick += 1

    xticklabels = list(data.keys())
    xticklabels_rotation = 0
    if 'xticklabels_rotation' in params:
        xticklabels_rotation = params['xticklabels_rotation']

    ax.set_xticks(list(range(len(data))))
    ax.set_xticklabels(xticklabels, fontsize=fontsize,
                       rotation=xticklabels_rotation)
    ax.set_ylabel(y1_label, fontsize=fontsize)
    ax2.set_ylabel(y2_label, fontsize=fontsize)
    if 'y1_ticks' in params:
        ax.set_yticks(params['y1_ticks'])
    if 'y2_ticks' in params:
        ax2.set_yticks(params['y2_ticks'])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label2.set_fontsize(fontsize)

    if display_legend:
        legend_params = {
            'loc': 'best',
            'frameon': False,
            'fontsize': fontsize
        }
        if 'legend' in params:
            legend_params.update(params['legend'])

        legend_separate = customized_params["legend_separate"] \
            if (customized_params and "legend_separate" in customized_params) \
            else False

    if display_legend and not legend_separate:
        label_params = ax.get_legend_handles_labels()
        label_params_2 = ax2.get_legend_handles_labels()
        merged_label_params = (
            label_params[0] + label_params_2[0],
            label_params[1] + label_params_2[1]
        )
        ax.legend(*merged_label_params, **legend_params)

    fig.savefig(
        fig_save_path,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if both_type:
        fig.savefig(
            fig_save_path + '.pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )

    if display_legend and legend_separate:
        legend_figure_params = {
            'figsize': (0.1, 0.1),
            'dpi': 1200
        }
        fig = plt.figure(**legend_figure_params)
        label_params = ax.get_legend_handles_labels()
        label_params_2 = ax2.get_legend_handles_labels()
        merged_label_params = (
            label_params[0] + label_params_2[0],
            label_params[1] + label_params_2[1]
        )
        axl = fig.add_subplot(111, label='a')
        axl.axis(False)
        axl.legend(*merged_label_params, **legend_params)
        fig.savefig(
            fig_save_path + '_legend',
            bbox_inches='tight',
            pad_inches=0.0
        )
        if both_type:
            fig.savefig(
                fig_save_path + '_legend.pdf',
                bbox_inches='tight',
                pad_inches=0.0
            )


def _time_sequence_plot(data, boundaries,
                        x_label, y_label, file_name,
                        label_info=None, y_ticklabels=None):
    _fontsize = 10
    fig = plt.figure(figsize=(1.8, 0.6), dpi=600)
    ax = fig.add_subplot(111)

    logical_y_list = list(data.keys())
    logical_y_list = sorted(logical_y_list)
    logical_y_list.reverse()
    num_y_ticklabels = len(logical_y_list)
    y_mapping = {}
    for idx, logical_y in enumerate(logical_y_list):
        y_mapping[logical_y] = idx + 1

    bar_label_to_plot = set()
    hatch_density = 2
    hatch_records = {}

    # to calculate utilization (for time_seq_view_2)
    resource_usage = {
        entity: 0 for entity in data.keys()
    }
    last_end_time = -1

    # print(file_name, data)
    for entity, entity_list in data.items():
        y = y_mapping[entity]
        for label_idx, x_start, x_end in entity_list:
            if label_info:
                bar_label = label_info[label_idx]["name"]
                type_idx = label_info[label_idx]["type_idx"]
                color = qualitative_colors[type_idx
                                           % len(qualitative_colors)]
                if type_idx not in hatch_records:
                    hatch_records[type_idx] = {
                        'cur': 0,
                        'labels': {}
                    }
                if label_idx \
                        not in hatch_records[type_idx]['labels']:
                    hatch_records[type_idx]['labels'][label_idx] \
                        = hatch_records[type_idx]['cur']
                    hatch_records[type_idx]['cur'] += 1
                hatch_idx = hatch_records[type_idx]['labels'][label_idx]
                hatch = bar_patterns[hatch_idx] * hatch_density
            else:
                bar_label = str(label_idx)
                color = qualitative_colors[label_idx]
                hatch=None
            if bar_label in bar_label_to_plot:
                bar_label = None
            else:
                bar_label_to_plot.add(bar_label)

            # to calculate utilization (for time_seq_view_2)
            resource_usage[entity] += x_end - x_start
            if x_end > last_end_time:
                last_end_time = x_end

            ax.barh(
                y=y, width=(x_end - x_start), left=x_start,
                align='center', label=bar_label, color=color,
                alpha=1.0, edgecolor='black', hatch=hatch
            )

    # to calculate utilization (for time_seq_view_2)
    # assuming there are only one round
    total_length = last_end_time - boundaries[0]
    # print(total_length, resource_usage,
    #       [1.0 - (a / total_length) for a in resource_usage.values()])
    # the last term is idle time

    for round_start_time in boundaries:
        ax.axvline(x=round_start_time, ymin=0, ymax=1,
                   color='#e0e0e0', dashes=[1, 1, 1, 1])

    ax.set_ylim((0, num_y_ticklabels + 1))
    ax.set_yticks(np.arange(1, num_y_ticklabels + 1))

    if y_ticklabels is None:
        y_ticklabels = list(data.keys())
    y_ticklabels.reverse()
    ax.set_yticklabels(y_ticklabels, fontsize=_fontsize)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=_fontsize)

    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(x_label, fontsize=_fontsize)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # TODO: for temporary use

    ncol = 2
    bbox_to_anchor = (0.4, 1.1)
    if len(labels) >= 3:
        ncol = 3
        nrow = (len(labels) - 1) // 3 + 1

        if nrow == 2:
            bbox_to_anchor = (0.4, 1.2)
        elif nrow == 3:
            bbox_to_anchor = (0.4, 1.3)
        elif nrow == 4:
            bbox_to_anchor = (0.4, 1.4)

    ax.legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc="center",
              ncol=ncol, frameon=False)
    # ax.legend(handles, labels, bbox_to_anchor=(1.0, 1.5), loc="center",
    #           ncol=(len(labels) - 1) // 5 + 1, frameon=False)
    fig.savefig(file_name, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0.0)


def time_sequence_plot(task_folder, result_dict, label_info=None):
    if not COORDINATOR in result_dict:
        return
    coordinator_dict = result_dict[COORDINATOR][ROUND]
    file_name = os.path.join(task_folder, output_rel)

    data = {}
    start_time_list = []
    data_2 = {}
    chunk_info = {}

    for round_idx, round_dict in coordinator_dict.items():
        # TODO: this is temporary chagne
        if round_idx > 0:
            break

        start_time = round_dict['overall']["start_time"]
        start_time_list.append(start_time)

        for chunk_idx, chunk_dict in round_dict[CHUNK].items():
            if chunk_idx not in data:
                data[chunk_idx] = []
            if chunk_idx not in chunk_info:
                chunk_info[chunk_idx] = {
                    'name': f"Chunk {chunk_idx}",
                    'type_idx': chunk_idx
                }

            for phase_idx, phase_dict in chunk_dict[PHASE].items():
                start_time = phase_dict["start_time"]
                end_time = phase_dict["end_time"]
                data[chunk_idx].append((phase_idx, start_time, end_time))

                resource_type = label_info[phase_idx]["type_idx"]
                if resource_type not in data_2:
                    data_2[resource_type] = []

                chunk_relationship = label_info[phase_idx]["chunk_relationship"]
                if not chunk_relationship == 0:  # "combine"
                    data_2[resource_type].append((
                        chunk_idx, start_time, end_time
                    ))

    chunk_info[len(chunk_info)] = {
        'name': f"all chunks",
        'type_idx': len(chunk_info)
    }

    sorted_keys = sorted(data.keys())
    sorted_data = {}
    for k in sorted_keys:
        sorted_data[k] = data[k]
    data = sorted_data

    _time_sequence_plot(
        data=data,
        boundaries=start_time_list,
        x_label="Time (s)",
        y_label="Chunk ID",
        file_name=file_name,
        label_info=label_info
    )

    _time_sequence_plot(
        data=data_2,
        boundaries=start_time_list,
        x_label="Time (s)",
        y_label="Resource",
        file_name=file_name + '_view_2',
        label_info=chunk_info,
        y_ticklabels=["c_comp", "comm", "s_comp"]
    )
