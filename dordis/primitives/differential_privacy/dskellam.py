import logging
import numpy as np
from dordis.config import Config
from dordis.primitives.differential_privacy.utils\
    .accounting_utils import dskellam_params
from dordis.primitives.differential_privacy.utils\
    .misc import clip_by_norm, scaled_quantization, \
    randomized_hadamard_transform, modular_clip_by_value, \
    inverse_scaled_quantization, inverse_randomized_hadamard_transform
from dordis.primitives.differential_privacy import base
from dordis.primitives.pseudorandom_generator import os_random


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
            'bits': Config().agg.differential_privacy.params.num_bits,
            'beta': np.exp(Config().agg.differential_privacy.params.log_beta),
            'l2_clip_norm': Config().agg.differential_privacy.params.l2_clip_norm,
            'k_stddevs': Config().agg.differential_privacy.params.k_stddevs,
        }

    def get_bits(self):
        return self.params_dict["bits"]

    def get_padded_dim(self, dim):
        return np.math.pow(2, np.ceil(np.log2(dim)))

    def init_params(self, dim, q, target_num_clients):
        logging.info(f"Initializing parameters for Dskellam...")
        padded_dim = self.get_padded_dim(dim=dim)

        # only do this for debugging (no pipeline)
        #         # self.params_dict.update({
        #         #     'target_num_clients': target_num_clients,
        #         #     'client_sampling_rate': q,
        #         #     'dim': dim,
        #         #     'padded_dim': padded_dim,
        #         #     'gamma': 0.0007840090435578685,
        #         #     'scale': 1275.4954910493823,
        #         #     'local_stddev': 2.6180339603380443,
        #         #     'local_scale': 3339.290511825333,
        #         # })
        #         # logging.info(f"Initialized parameters "
        #         #              f"for DSkellam: {self.params_dict}.")
        #         # return self.params_dict

        # self.params_dict.update({
        #     'target_num_clients': target_num_clients,
        #     'client_sampling_rate': q,
        #     'dim': dim,
        #     'padded_dim': padded_dim,
        #     'gamma': 0.0002011437157919468,
        #     'scale': 4971.56968619567,
        #     'local_stddev': 0.7834678476952787,
        #     'local_scale': 3895.0650017108137,
        # })
        # logging.info(f"Initialized parameters "
        #              f"for DSkellam: {self.params_dict}.")
        # return self.params_dict

        if hasattr(Config().clients, "attending_rate_upperbound"):
            attending_rate_upperbound = Config().clients.attending_rate_upperbound
            actual_steps = int(np.floor(attending_rate_upperbound * self.num_rounds))
        elif hasattr(Config().clients, "attending_time_upperbound"):
            actual_steps = Config().clients.attending_time_upperbound
        else:
            actual_steps = self.num_rounds

        scale, local_stddev = dskellam_params(
            q=q,
            dim=padded_dim,
            steps=actual_steps,
            num_clients=target_num_clients,
            bits=self.params_dict["bits"],
            delta=self.params_dict["delta"],
            beta=self.params_dict["beta"],
            k=self.params_dict["k_stddevs"],
            epsilon=self.params_dict["epsilon"],
            l2_clip_norm=self.params_dict["l2_clip_norm"],
        )
        gamma = 1.0 / scale

        self.params_dict.update({
            'target_num_clients': target_num_clients,
            'client_sampling_rate': q,
            'dim': dim,
            'padded_dim': padded_dim,
            'gamma': gamma,
            'scale': scale,
            'local_stddev': local_stddev,
            'local_scale': local_stddev * scale,
            'steps': actual_steps
        })

        logging.info(f"Initialized parameters "
                     f"for DSkellam: {self.params_dict}.")
        return self.params_dict

    def add_local_noise(self, record, local_stddev, seed=None, subtract=False):
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

    def encode_data(self, data, log_prefix_str, other_args):
        sample_hadamard_seed, num_sampled_clients, dp_params_dict, \
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

        # Flatten
        data = randomized_hadamard_transform(
            data=data,
            seed=sample_hadamard_seed
        )
        logging.info("%s Data rotated.", log_prefix_str)

        # Conditional randomized round
        scale = dp_params_dict["scale"]
        beta = dp_params_dict["beta"]
        logging.info(f"[Debug] before scale {round(scale, 3)}: norm: {round(np.linalg.norm(data), 3)}, "
                     f"max: {round(max(data), 3)}, min: {round(min(data), 3)}, "
                     f"first 6: {[round(e, 3) for e in data[:6]]}, "
                     f"last 6: {[round(e, 3) for e in data[-6:]]}.")
        data = scaled_quantization(
            data=data,
            scale=scale,
            stochastic=True,
            conditional=True,
            l2_norm_bound=l2_clip_norm,
            beta=beta
        )
        logging.info(f"[Debug] after scale: norm: {round(np.linalg.norm(data), 3)}, "
                     f"max: {round(max(data), 3)}, min: {round(min(data), 3)}, "
                     f"first six: {[round(e, 3) for e in data[:6]]}, "
                     f"last six: {[round(e, 3) for e in data[-6:]]}.")
        logging.info("%s Data rounded.", log_prefix_str)

        # Add noise
        local_stddev = dp_params_dict["local_stddev"]
        target_num_clients = dp_params_dict["target_num_clients"]
        central_stddev = np.sqrt(target_num_clients) * local_stddev

        actual_target_num_clients = num_sampled_clients
        if hasattr(Config().agg.differential_privacy, "pessimistic"):
            dropout_frac_tolerated = Config().agg.differential_privacy.pessimistic
            actual_target_num_clients *= (1 - dropout_frac_tolerated)
            # This assume that the client selection ensures that
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

        # Modular clipping
        bits = dp_params_dict["bits"]
        mod_clip_lo, mod_clip_hi = -(2 ** (bits - 1)), 2 ** (bits - 1)
        data = modular_clip_by_value(
            data=data,
            clip_range_lower=mod_clip_lo,
            clip_range_upper=mod_clip_hi
        )
        logging.info("%s Modular clipped.", log_prefix_str)

        return data.tolist(), excessive_noise_seeds

    def decode_data(self, data, log_prefix_str, other_args):
        data = np.array(data)
        sample_hadamard_seed, dp_params_dict = other_args

        # logging.info(f"[Debug] {type(data)} {data}.")

        scale = dp_params_dict["scale"]
        data = inverse_scaled_quantization(
            data=data,
            scale=scale
        )
        logging.info("%s Modular dequantized.", log_prefix_str)

        data = inverse_randomized_hadamard_transform(
            data=data,
            seed=sample_hadamard_seed
        )
        logging.info("%s Data rotated.", log_prefix_str)
        return data.tolist()
