import numpy as np
import logging
from hyades.config import Config

DEFAULT_BETA = np.exp(-0.5)


def clip_by_norm(data, l2_clip_norm):
    norm = np.linalg.norm(data)
    logging.info(f"[Debug] Norm before clipping: {norm}.")
    if norm > l2_clip_norm:
        logging.info(f"Data before clipping: {data[:3]} {data[-3:]}.")
        data = data * l2_clip_norm / norm
        logging.info(f"Data after clipping: {data[:3]} {data[-3:]}.")
    return data


def stochastic_rounding(data, conditional,
                        l2_norm_bound=None, beta=DEFAULT_BETA):

    def post_rounding_l2_norm_bound(data, l2_norm_bound, beta):
        dim = data.size
        if l2_norm_bound is None:
            data_norm = np.linalg.norm(data)
        else:
            data_norm = l2_norm_bound

        bound1 = data_norm + np.sqrt(dim)
        squared_bound2 = np.square(data_norm) + 0.25 * dim
        squared_bound2 += np.sqrt(2.0 * np.math.log(1.0 / beta)) * \
                          (data_norm + 0.5 * np.sqrt(dim))
        bound2 = np.sqrt(squared_bound2)
        return min(bound1, bound2)

    l2_norm_threshold = post_rounding_l2_norm_bound(data, l2_norm_bound, beta)
    floored_data = np.floor(data)
    decimal_x = data - floored_data

    rounded_data = floored_data
    rounded_l2_norm = l2_norm_threshold + 1
    while conditional and rounded_l2_norm > l2_norm_threshold:
        uniform = np.random.random(data.shape)
        bernoulli = (uniform < decimal_x).astype(float)
        rounded_data = floored_data + bernoulli
        rounded_l2_norm = np.linalg.norm(rounded_data)

    return rounded_data


def scaled_quantization(data, scale, stochastic=True,
                        l2_norm_bound=None, beta=DEFAULT_BETA,
                        conditional=True):
    scaled_data = data * scale
    scaled_bound = l2_norm_bound * scale

    if stochastic:
        quantized_data = stochastic_rounding(
            data=scaled_data,
            conditional=conditional,
            l2_norm_bound=scaled_bound,
            beta=beta
        )
    else:
        quantized_data = np.round(scaled_data)

    return quantized_data.astype(int)


def inverse_scaled_quantization(data, scale):
    return data / scale


def pad_zeros(data):
    dim = data.size
    log2_dim = np.log2(dim)
    pad_dim = np.power(2, int(np.ceil(log2_dim)))
    return np.pad(data, (0, pad_dim - dim))


def sample_rademacher(shape, seed):
    np.random.seed(seed)
    rand_uniform = np.random.random(size=shape)
    return np.sign(rand_uniform - 0.5)


def fast_walsh_hadamard_transform(data):
    original_data_shape = data.shape
    assert len(original_data_shape) == 2

    dim = original_data_shape[-1]
    if not (dim and ((dim & (dim - 1)) == 0)):
        raise ValueError(f'The dimension of data must be a power of 2. '
                         f'Provided dimension is {dim}')
    log2 = int(np.ceil(np.log2(dim)))
    if dim == 1:
        return np.identity(data)

    h_core = np.array([[1., 1.], [1., -1.]])
    permutation = np.array([0, 2, 1])

    def _hadamard_step(data, dim):
        data_shape = data.shape
        data = data.reshape((-1, 2))
        data = np.matmul(data, h_core)
        data = data.reshape((-1, dim // 2, 2))
        data = np.transpose(data, permutation)
        data = data.reshape(data_shape)
        return data

    def _fwht(data, dim, log2):
        data = data.reshape((-1, 2, dim // 2))
        i = 0
        while i < log2:
            data = _hadamard_step(data, dim)
            i += 1
        return data

    data = _fwht(data, dim, log2)
    data = data.reshape((-1, dim))
    data /= np.sqrt(dim)
    data = data.reshape(original_data_shape)
    return data


def randomized_hadamard_transform(data, seed, repeat=1):
    padded_data = pad_zeros(data)

    i = 0
    while i < repeat:
        cur_seed = seed + i
        signs = sample_rademacher(padded_data.size, seed=cur_seed)
        rademacher_data = signs * padded_data
        padded_data = np.squeeze(fast_walsh_hadamard_transform(
            data=np.expand_dims(rademacher_data, axis=0)
        ), axis=0)
        i += 1

    return padded_data


def inverse_randomized_hadamard_transform(data, seed, repeat=1):

    i = repeat - 1
    while i >= 0:
        cur_seed = seed + i
        unrotated_data = fast_walsh_hadamard_transform(
            np.expand_dims(data, axis=0)
        )
        unrotated_data = np.squeeze(unrotated_data, axis=0)
        signs = sample_rademacher(
            shape=unrotated_data.shape,
            seed=cur_seed
        )
        data = signs * unrotated_data
        i -= 1

    return data


def modular_clip_by_value(data, clip_range_lower, clip_range_upper):
    width = clip_range_upper - clip_range_lower
    period = np.floor(data / width - clip_range_lower / width).astype(int)
    data_mod_clipped = data - period * width
    return data_mod_clipped


def modular_clip(data, log_prefix_str, other_args):
    data = np.array(data)
    discrete_params_dict, = other_args
    bits = discrete_params_dict["bits"]
    mod_clip_lo, mod_clip_hi = -(2 ** (bits - 1)), 2 ** (bits - 1)
    data = modular_clip_by_value(
        data=data,
        clip_range_lower=mod_clip_lo,
        clip_range_upper=mod_clip_hi
    )
    logging.info("%s Modular clipped.", log_prefix_str)

    return data.tolist()
