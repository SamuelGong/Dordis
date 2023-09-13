import psutil
import logging
import numpy as np
from dordis.config import Config


class CPUAffinity:
    def __init__(self):
        self.cpu_affinity_dict = None

        if hasattr(Config().agg, "cpu_affinity"):
            self.cpu_affinity_dict = {}

            type = Config().agg.cpu_affinity.type
            if type == "simple":
                args = Config().agg.cpu_affinity.args
                comm_portion = args[0]
                assert 0 < comm_portion < 1

                cpu_list = sorted(list(range(psutil.cpu_count())))
                num_cpus_for_comm = int(np.ceil(comm_portion * len(cpu_list)))
                assert 0 < num_cpus_for_comm < len(cpu_list)

                self.cpu_affinity_dict['comm'] = cpu_list[-num_cpus_for_comm:]
                self.cpu_affinity_dict['comp'] = cpu_list[:-num_cpus_for_comm]

            logging.info(f"cpu_affinity_dict: {self.cpu_affinity_dict}.")

    def set_cpu_affinity(self, affinity):
        proc = psutil.Process()
        proc.cpu_affinity(affinity)

        logging.info(f"Process {proc.pid}'s affinity "
                     f"set to {affinity}.")

        # before = proc.cpu_affinity()
        # proc.cpu_affinity(affinity)
        # after = proc.cpu_affinity()
        #
        # logging.info(f"Process {proc.pid}'s affinity "
        #              f"has been set from {before} to {after}.")
