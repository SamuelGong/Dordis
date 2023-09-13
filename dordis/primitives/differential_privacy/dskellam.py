import logging
import numpy as np
from dordis.config import Config
from dordis.primitives.differential_privacy.utils\
    .skellam import add_local_noise
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
        self.num_rounds = Config().app.repeat
        if hasattr(Config().agg.differential_privacy.params, "num_rounds"):
            self.num_rounds = Config().agg.differential_privacy.params.num_rounds
        else:
            self.num_rounds = Config().app.repeat
        self.pseudorandom_generator = os_random.Handler()

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
        self.xnoise_params = {}

    def get_bits(self):
        return self.params_dict["bits"]

    def get_padded_dim(self, dim):
        return np.math.pow(2, np.ceil(np.log2(dim)))

    def get_xnoise_params(self, round_idx, num_sampled_clients,
                          dp_params_dict):
        local_stddev = dp_params_dict["local_stddev"]

        # this is needed for compatibility with trace-driven
        target_num_clients = dp_params_dict["target_num_clients"]
        requried_central_stddev = local_stddev * np.sqrt(target_num_clients)
        actual_baseline_local_stddev = requried_central_stddev / np.sqrt(num_sampled_clients)

        if round_idx not in self.xnoise_params:
            dropout_tolerated_frac \
                = Config().agg.differential_privacy\
                .dropout_resilience.dropout_tolerated_frac

            dropout_tolerated = int(np.floor(
                num_sampled_clients * dropout_tolerated_frac))
            method_type = Config().agg.differential_privacy \
                .dropout_resilience.type

            d = {}
            component_stddevs = []
            if method_type == "simple":
                num_seeds = dropout_tolerated
                for i in range(num_seeds):
                    j = dropout_tolerated - i
                    stddev_i = actual_baseline_local_stddev * np.sqrt(
                        num_sampled_clients
                        / (num_sampled_clients - j)
                        / (num_sampled_clients - j + 1)
                    )
                    component_stddevs.append(stddev_i)
            elif method_type == "log2":
                num_noise_levels = int(np.ceil(np.log2(dropout_tolerated)))
                num_seeds = num_noise_levels + 1

                noise_max_var = dropout_tolerated \
                                / (num_sampled_clients - dropout_tolerated) \
                                * actual_baseline_local_stddev ** 2
                noise_min_var = noise_max_var / 2 ** num_noise_levels
                for i in range(0, num_noise_levels):
                    component_stddevs.append(np.sqrt(2 ** i * noise_min_var))
                component_stddevs = [component_stddevs[0]] + component_stddevs

                d.update({
                    "dropout_tolerated": dropout_tolerated,
                    "noise_min_var": noise_min_var,
                    "num_noise_levels": num_noise_levels
                })
            else:
                raise ValueError(f"Dropout resilience: unknown type: {method_type}.")

            d.update({
                "num_seeds": num_seeds,
                "component_stddevs": component_stddevs
            })
            self.xnoise_params[round_idx] = d

        return self.xnoise_params[round_idx]

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

        data = add_local_noise(
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

    def add_excessive_noise(self, data, xnoise_params,
                            log_prefix_str, seeds=None, subtract=False):
        if hasattr(Config().agg.differential_privacy, "dropout_resilience"):
            if not seeds:  # generate seeds on her own
                if subtract:
                    return data, None

                num_seeds = xnoise_params["num_seeds"]
                seeds = self.generate_excessive_noise_seeds(num_seeds)
                component_stddevs = xnoise_params["component_stddevs"]
                seeds = list(zip(seeds, component_stddevs))

            if subtract:
                logging.info("%s Start subtracting %d excessive noise.",
                             log_prefix_str, len(seeds))
            else:
                logging.info("%s Start adding %d excessive noise.",
                             log_prefix_str, len(seeds))

            if not isinstance(data, np.ndarray):
                data = np.array(data)
            for i, tu in enumerate(seeds):
                data = add_local_noise(
                    record=data,
                    local_stddev=tu[1],
                    seed=tu[0],
                    subtract=subtract
                )
            if subtract:
                # logging.info(f"[Debug] After subtracting execessive noise: "
                #              f"first six: {data[:6]}, "
                #              f"last six: {data[-6:]}.")
                logging.info("%s Excessive noise seeded by %s "
                             "is subtracted.",
                             log_prefix_str, seeds)
            else:
                # logging.info(f"[Debug] After adding execessive noise: "
                #              f"first six: {data[:6]}, "
                #              f"last six: {data[-6:]}.")
                logging.info("%s Excessive noise seeded by %s "
                             "is added for dropout resilience.",
                             log_prefix_str, seeds)

            return data, seeds
        else:
            return data, None

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

    def generate_excessive_noise_seeds(self, num_seeds):
        seeds = self.pseudorandom_generator \
            .generate_numbers(
            num_range=(0, 1 << 32),  # 32 bits each
            dim=num_seeds
        )

        return seeds
