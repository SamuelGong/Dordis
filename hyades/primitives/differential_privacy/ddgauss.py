import logging
import numpy as np
from hyades.primitives.differential_privacy.utils\
    .dgauss import add_local_noise
from hyades.primitives.differential_privacy.utils\
    .accounting_utils import ddgauss_params
from hyades.primitives.differential_privacy.utils\
    .misc import clip_by_norm, scaled_quantization, \
    randomized_hadamard_transform, modular_clip_by_value, \
    inverse_scaled_quantization, inverse_randomized_hadamard_transform
from hyades.primitives.differential_privacy import base
from hyades.config import Config


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
        padded_dim = self.get_padded_dim(dim=dim)

        # if Config().agg.type == "dp_plus_secagg" \
        #         and hasattr(Config().agg.differential_privacy, "pessimistic"):
        #     dropout_frac_tolerated = Config().agg.differential_privacy.pessimistic
        #     target_num_clients = int(np.floor(target_num_clients
        #                                         * (1 - dropout_frac_tolerated)))

        if hasattr(Config().clients, "attending_rate_upperbound"):
            attending_rate_upperbound = Config().clients.attending_rate_upperbound
            actual_steps = int(np.floor(attending_rate_upperbound * self.num_rounds))
        elif hasattr(Config().clients, "attending_time_upperbound"):
            actual_steps = Config().clients.attending_time_upperbound
        else:
            actual_steps = self.num_rounds

        gamma, local_stddev = ddgauss_params(
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
        scale = 1.0 / gamma

        self.params_dict.update({
            'dim': dim,
            'padded_dim': padded_dim,
            'gamma': gamma,
            'scale': scale,
            'local_stddev': local_stddev,
            'local_scale': local_stddev * scale,
            'target_num_clients': target_num_clients,
            'client_sampling_rate': q,
            'steps': actual_steps
        })

        logging.info(f"Initialized parameters "
                     f"for DDGauss: {self.params_dict}.")
        return self.params_dict

    def encode_data(self, data, log_prefix_str, other_args):
        sample_hadamard_seed, _, dp_params_dict, \
            full_data_norm, _ = other_args
        # _: just to be consistent with DSkellam
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
        data = scaled_quantization(
            data=data,
            scale=scale,
            stochastic=True,
            conditional=True,
            l2_norm_bound=l2_clip_norm,
            beta=beta
        )
        logging.info("%s Data rounded.", log_prefix_str)

        # Add noise
        local_scale = dp_params_dict["local_scale"]
        data = add_local_noise(
            record=data,
            local_scale=local_scale
        )
        logging.info("%s Noise added.", log_prefix_str)

        # Modular clipping
        bits = dp_params_dict["bits"]
        mod_clip_lo, mod_clip_hi = -(2 ** (bits - 1)), 2 ** (bits - 1)
        data = modular_clip_by_value(
            data=data,
            clip_range_lower=mod_clip_lo,
            clip_range_upper=mod_clip_hi
        )
        logging.info("%s Modular clipped.", log_prefix_str)

        return data.tolist(), None

    def decode_data(self, data, log_prefix_str, other_args):
        data = np.array(data)
        sample_hadamard_seed, discrete_params_dict = other_args

        scale = discrete_params_dict["scale"]
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
