import logging
import numpy as np
from hyades.config import Config
from hyades.utils.share_memory_handler \
    import ShareBase, CLIENT_STATS, \
    SIMULATED_CLIENT_LATENCY, CLIENT_ATTENDANCE


class ClientManager(ShareBase):
    def __init__(self) -> None:
        ShareBase.__init__(self, client_id=0)
        self.total_clients = Config().clients.total_clients
        self.num_rounds = Config().app.repeat

        if hasattr(Config().clients, "attending_rate_upperbound") \
                or hasattr(Config().clients, "attending_time_upperbound"):
            if hasattr(Config().clients, "attending_rate_upperbound"):

                self.attending_rate_upperbound = Config().clients.attending_rate_upperbound
                self.attending_max_rounds = int(np.floor(self.num_rounds
                                                         * self.attending_rate_upperbound))
            else:
                self.attending_max_rounds = Config().clients.attending_time_upperbound

            initialized_attendance_record = {
                client_id: 0 for client_id in range(1, self.total_clients + 1)
            }
            self.set_a_shared_value(
                key=[CLIENT_ATTENDANCE],
                value=initialized_attendance_record
            )

            logging.info(f"[Client Manager] Client maximum attendance: "
                         f"{self.attending_max_rounds}.")
        else:
            self.attending_max_rounds = None

        self.init_client_latency()

    def set_client_dataset_size(self, client_dataset_size_dict):
        # so that other process (e.g., client sampler) can use
        for client_id, dataset_size in client_dataset_size_dict.items():
            temp = self.get_a_shared_value(
                key=[CLIENT_STATS, client_id]
            )
            temp["training_dataset_size"] = dataset_size
            self.set_a_shared_value(
                key=[CLIENT_STATS, client_id],
                value=temp
            )

    def init_client_latency(self):
        # visiable to client selector
        for client_id in range(1, self.total_clients + 1):
            self.set_a_shared_value(
                key=[CLIENT_STATS, client_id],
                value={
                    "round_stats": {
                        -1: {
                            "time": 1,  # initialized value
                            "used": False
                        }
                    }
                }
            )

        # for simulation, not visible to client selector
        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple" \
                and hasattr(Config().simulation, "time"):
            type = Config().simulation.time.type
            client_latency = {}
            if type == "zipf":
                from hyades.utils.misc import my_random_zipfian

                a = Config().simulation.time.a
                seed = Config().simulation.time.seed
                min = Config().simulation.time.min
                max = Config().simulation.time.max

                temp = my_random_zipfian(
                    a=a,
                    n=self.total_clients,
                    amin=min,
                    amax=max,
                    seed=seed
                )
                prompt_prefix = f"[Client Manager] Client latencies initialized " \
                                f"(zipf, a={a}, seed={seed}, "\
                         f"range: [{min}, {max}]), client_latencies: "
            elif type == "uniform":
                seed = Config().simulation.time.seed
                min = Config().simulation.time.min
                max = Config().simulation.time.max
                np.random.seed(seed)

                rng = np.random.default_rng(seed=seed)
                temp = rng.uniform(min, max, self.total_clients)
                rng.shuffle(temp)
                temp = temp.tolist()
                prompt_prefix = f"[Client Manager] Client latencies initialized " \
                                f"(uniform, seed={seed}, " \
                         f"range: [{min}, {max}]), client_latencies: "
            else:
                raise NotImplementedError

            for idx, value in enumerate(temp):
                client_latency[idx + 1] = value

            for client_id, latency in client_latency.items():
                self.set_a_shared_value(
                    key=[SIMULATED_CLIENT_LATENCY, client_id],
                    value=latency
                )
            prompt = prompt_prefix + f"{client_latency}."
            logging.info(prompt)

    def get_available_clients(self):
        if self.attending_max_rounds:
            attendance_record = self.get_a_shared_value(
                key=CLIENT_ATTENDANCE
            )
            return [client_id for client_id, attended_rounds in attendance_record.items()
                    if attended_rounds < self.attending_max_rounds]
        else:  # no constraint
            return list(range(1, self.total_clients + 1))
