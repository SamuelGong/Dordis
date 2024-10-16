import logging
import numpy as np
from dordis.config import Config
from dordis.primitives.differential_privacy import base
from opacus.accountants.utils import get_noise_multiplier
from dordis.primitives.pseudorandom_generator import os_random
from dordis.primitives.differential_privacy.utils\
    .misc import clip_by_norm


class Handler(base.Handler):
    def __init__(self):
        super().__init__()
        if hasattr(Config().agg.differential_privacy.params, "num_rounds"):
            self.num_rounds = Config().agg.differential_privacy.params.num_rounds
        else:
            self.num_rounds = Config().app.repeat

        delta = Config().agg.differential_privacy.params.delta \
            if hasattr(Config().agg.differential_privacy.params, "delta") \
            else 1. / Config().clients.total_clients
        self.params_dict = {
            'epsilon': Config().agg.differential_privacy.params.epsilon,
            'delta': delta,
            'l2_clip_norm': Config().agg.differential_privacy.params.l2_clip_norm,
        }

    def init_params(self, dim, q, target_num_clients):
        # dim is not used in this mechanism
        logging.info(f"Initializing parameters for Gaussian...")

        if hasattr(Config().clients, "attending_rate_upperbound"):
            attending_rate_upperbound = Config().clients.attending_rate_upperbound
            actual_steps = int(np.floor(attending_rate_upperbound * self.num_rounds))
        elif hasattr(Config().clients, "attending_time_upperbound"):
            actual_steps = Config().clients.attending_time_upperbound
        else:
            actual_steps = self.num_rounds

        # reference: https://github.com/pytorch/opacus/blob/main/opacus/accountants/utils.py
        noise_multiplier = get_noise_multiplier(
            target_epsilon=self.params_dict["epsilon"],
            target_delta=self.params_dict["delta"],
            sample_rate=q,
            steps=actual_steps,
            accountant="rdp"
        )

        # noise_multiplier = global_stddev / l2_clip_norm
        central_stddev = self.params_dict["l2_clip_norm"] * noise_multiplier
        # because guassian noise is additive
        local_stddev = central_stddev / np.sqrt(target_num_clients)

        self.params_dict.update({
            'noise_multiplier': noise_multiplier,
            'central_stddev': central_stddev,
            'local_stddev': local_stddev,
            'target_num_clients': target_num_clients,
        })
        logging.info(f"Initialized parameters "
                     f"for Gaussian: {self.params_dict}.")
        return self.params_dict

    def add_local_noise(self, record, local_stddev, seed=None, subtract=False):
        if seed:
            np.random.seed(seed)
        else:
            pseudorandom_generator = os_random.Handler()
            seed = pseudorandom_generator.generate_numbers((0, 1 << 32), 1)[0]
            np.random.seed(seed)

        noise = np.random.normal(0, local_stddev, record.shape)
        if subtract:
            return record - noise
        else:
            return record + noise

    def encode_data(self, data, log_prefix_str, other_args):
        _, num_sampled_clients, dp_params_dict, \
            full_data_norm, xnoise_params = other_args
        data = np.array(data)

        # Clip
        l2_clip_norm = dp_params_dict["l2_clip_norm"]
        norm = np.linalg.norm(data)
        proper_l2_clip_norm = norm / full_data_norm * l2_clip_norm
        data = clip_by_norm(
            data=data,
            l2_clip_norm=proper_l2_clip_norm
        )
        logging.info("%s Data clipped to have norm %.4f.",
                     log_prefix_str, proper_l2_clip_norm)

        # Add noise
        local_stddev = dp_params_dict["local_stddev"]
        central_stddev = dp_params_dict["central_stddev"]
        actual_target_num_clients = num_sampled_clients

        if hasattr(Config().agg.differential_privacy, "pessimistic"):
            dropout_frac_tolerated = Config().agg.differential_privacy.pessimistic
            actual_target_num_clients *= (1 - dropout_frac_tolerated)
            # This assumes that the client selection ensures that
            # at least 1 client survives
            actual_target_num_clients = max(1, int(np.floor(actual_target_num_clients)))

        actual_local_stddev = central_stddev / np.sqrt(actual_target_num_clients)
        logging.info(f"local_stddev calculated before training: {local_stddev}, "
                     f"actual_local_stddev: {actual_local_stddev}.")

        data = self.add_local_noise(
            record=data,
            local_stddev=actual_local_stddev
        )
        logging.info(f"{log_prefix_str} Noise added "
                     f"with local_stddev {round(actual_local_stddev, 4)}.")
        # for dropout resilience
        data, excessive_noise_seeds = self.add_excessive_noise(
            data=data,
            xnoise_params=xnoise_params,
            log_prefix_str=log_prefix_str
        )

        return data.tolist(), excessive_noise_seeds
