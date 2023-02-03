import numpy as np
import multiprocessing as mp
from hyades.utils.misc import get_chunks_idx

# N_JOBS = min(mp.cpu_count(), 1)
# N_JOBS = mp.cpu_count()
N_JOBS = 1


def _asymmetric_quantize(array, clipping_range, bit_width):
    res = np.clip(array, clipping_range[0], clipping_range[1])  # clip
    res = (res - clipping_range[0]) \
          / (clipping_range[1] - clipping_range[0]) \
          * ((1 << bit_width) - 1) # linear transform
    res = np.floor(res + np.random.random(len(res)))  # stochastic round
    res = res.astype(int).astype(object)  # np.float to np.int to int
    res = res.tolist()  # enable concatenation
    return res


def _asymmetric_unquantize(array, clipping_range, bit_width, l):
    res = np.array(array) / (((1 << bit_width) - 1) * l) \
          * ((clipping_range[1] - clipping_range[0]) * l) \
          + clipping_range[0] * l # linear transform
    res = res.astype(object)  # np.float to float
    res = res.tolist()  # enable concatenation
    return res


def quantize(flatten_array, params, l=None, padded_num_bits=0):
    quantization_type = params.type
    if quantization_type == "asymmetric":
        clipping_range = params.clipping_range
        bit_width = params.bit_width - padded_num_bits
        if l is None:  # quantize
            worker_function = _asymmetric_quantize
            other_args = [clipping_range, bit_width]
        else:  # unquantize
            worker_function = _asymmetric_unquantize
            other_args = [clipping_range, bit_width, l]
    else:
        raise ValueError(f"No such quantization type "
                         f"{quantization_type}.")

    pool_inputs = []
    for begin, end in get_chunks_idx(
        l=len(flatten_array),
        n=N_JOBS
    ):
        pool_inputs.append([flatten_array[begin:end]] + other_args)

    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    with mp.Pool(N_JOBS) as pool:
        pool_outputs = pool.starmap(worker_function, pool_inputs)

    res = []
    for arr in pool_outputs:
        res += arr

    return res
