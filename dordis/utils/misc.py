import time
import numpy as np
from functools import reduce


def mocking_cpu_intensive():
    _ = np.random.randint(low=0, high=10000, size=10000)


def get_chunks_idx(l, n):
    d, r = divmod(l, n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def get_chunks_idx_with_mod(l, n, mod):
    group_num = (l - 1) // mod + 1
    d, r = divmod(group_num, n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        begin = si * mod
        end = (si + (d + 1 if i < r else d)) * mod
        if end > l:
            end = l
        yield begin, end


def plaintext_add(x, y, mod_bit=None):
    # x, y may be a list instead of nd.array
    if mod_bit is None:
        return [xi + yi for xi, yi in zip(x, y)]
    else:
        mod_mask = (1 << mod_bit) - 1
        return [(xi + yi) & mod_mask for xi, yi in zip(x, y)]


def plaintext_aggregate(data_list, mod_bit=None):
    return reduce(lambda x, y: plaintext_add(x, y, mod_bit), data_list)


def calc_sleep_time(sec_per_step, cur_step, start_time, gap=0):
    expected_time = sec_per_step * cur_step
    actual_time = time.perf_counter() - start_time
    start_time_drift = actual_time - expected_time
    sleep_time = max(gap, sec_per_step - start_time_drift)
    return sleep_time


def my_random_zipfian(a, n, amin, amax, seed=None):
    prob = np.array([1 / k**a for k
                     in np.arange(1, n + 1)])
    res = [(e - min(prob)) / (max(prob) - min(prob)) * (amax - amin) + amin for e in prob]
    res = [round(e, 2) for e in res]

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(res)
    else:
        np.random.shuffle(res)
    return res
