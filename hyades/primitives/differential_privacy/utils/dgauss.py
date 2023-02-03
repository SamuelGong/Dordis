import numpy as np
from hyades.primitives.pseudorandom_generator import os_random


def _sample_discrete_laplace(t, shape):
    geometric_probs = 1.0 - np.exp(-1.0 / t)
    geo1 = np.random.geometric(p=geometric_probs, size=shape)
    geo2 = np.random.geometric(p=geometric_probs, size=shape)
    return np.int64(geo1 - geo2)


def _sample_bernoulli(p):
    return np.random.binomial(n=1, p=p)


def sample_discrete_gaussian(scale, shape, dtype=int):
    sq_scale = scale ** 2
    dlap_scale = scale
    oversample_factor = 1.5

    min_n = 1000
    target_n = int(np.prod(shape))
    oversample_n = int(oversample_factor * target_n)
    draw_n = max(min_n, oversample_n)

    accepted_n = 0
    result = np.zeros((0,), dtype=int)
    while accepted_n < target_n:
        samples = _sample_discrete_laplace(dlap_scale, shape=(draw_n,))
        z_numer = (np.abs(samples) - scale) ** 2
        z_denom = 2 * sq_scale
        bern_probs = np.exp(-z_numer/z_denom)
        accept = _sample_bernoulli(bern_probs)
        accepted_samples = samples[np.equal(accept, 1)]
        accepted_n += np.size(accepted_samples)
        result = np.concatenate([result, accepted_samples], axis=0)
        draw_n = (target_n - accepted_n) * oversample_factor
        draw_n = max(min_n, int(draw_n))

    return result[:target_n].reshape(shape).astype(dtype)


def add_local_noise(record, local_scale, shares=1, seed=None):
    ceil_local_scale = int(np.ceil(local_scale))
    shape = np.concatenate([[shares], record.shape], axis=0)

    if seed:
        np.random.seed(seed)
    else:
        pseudorandom_generator = os_random.Handler()
        seed = pseudorandom_generator.generate_numbers((0, 1 << 32), 1)[0]
        np.random.seed(seed)

    dgauss_noise = sample_discrete_gaussian(
        scale=ceil_local_scale, shape=shape, dtype=record.dtype
    )
    # print(ceil_local_scale, dgauss_noise)
    return record + np.sum(dgauss_noise, axis=0)
