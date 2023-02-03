import logging

import numpy as np
from hyades.primitives.pseudorandom_generator import os_random


def add_local_noise(record, local_stddev, seed=None, subtract=False):
    poisson_lam = 0.5 * local_stddev * local_stddev
    if seed:
        np.random.seed(seed)
    else:
        pseudorandom_generator = os_random.Handler()
        seed = pseudorandom_generator.generate_numbers((0, 1 << 32), 1)[0]
        np.random.seed(seed)
    poisson_1 = np.random.poisson(poisson_lam, record.shape)
    poisson_2 = np.random.poisson(poisson_lam, record.shape)

    # reserved for gaining some sense when determining clipping bound
    # but do not want to overwhelm the server's log
    if not subtract:
        logging.info(f"[Debug] Skellam: "
                     f"norm: {np.linalg.norm(poisson_1 - poisson_2)}, "
                     f"max: {max(poisson_1 - poisson_2)}, "
                     f"min: {min(poisson_1 - poisson_2)}.")
        logging.info(f"[Debug] First 6: record: {record[:6]}, "
                     f"skellam: {poisson_1[:6] - poisson_2[:6]}.")
        logging.info(f"[Debug] Last 6: record: {record[-6:]}, "
                     f"skellam: {poisson_1[-6:] - poisson_2[-6:]}.")

    if subtract:
        return record - poisson_1 + poisson_2
    else:
        return record + poisson_1 - poisson_2
