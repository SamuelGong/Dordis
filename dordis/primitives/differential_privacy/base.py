import logging
import numpy as np
from abc import abstractmethod
from dordis.config import Config
from dordis.primitives.pseudorandom_generator import os_random


class Handler:
    def __init__(self):
        self.total_clients = Config().clients.total_clients
        self.xnoise_params = {}
        self.pseudorandom_generator = os_random.Handler()

    @abstractmethod
    def init_params(self, dim, q, target_num_clients):
        """ """

    @abstractmethod
    def encode_data(self, data, log_prefix_str, other_args):
        """ """

    @abstractmethod
    def decode_data(self, data, log_prefix_str, other_args):
        """ """

    @abstractmethod
    def get_bits(self):
        """ """

    @abstractmethod
    def get_padded_dim(self, dim):
        """ """

    @abstractmethod
    def add_local_noise(self, record, local_stddev, seed=None, subtract=False):
        """ """

    def get_xnoise_params(self, round_idx, num_sampled_clients,
                          dp_params_dict):
        local_stddev = dp_params_dict["local_stddev"]

        # this is needed for compatibility with trace-driven
        target_num_clients = dp_params_dict["target_num_clients"]
        requried_central_stddev = local_stddev * np.sqrt(target_num_clients)
        actual_baseline_local_stddev = requried_central_stddev / np.sqrt(num_sampled_clients)

        if round_idx not in self.xnoise_params:
            dropout_tolerated_frac \
                = Config().agg.differential_privacy \
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
                data = self.add_local_noise(
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

    def generate_excessive_noise_seeds(self, num_seeds):
        seeds = self.pseudorandom_generator \
            .generate_numbers(
            num_range=(0, 1 << 32),  # 32 bits each
            dim=num_seeds
        )

        return seeds
