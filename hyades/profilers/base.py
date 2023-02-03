import logging
import time
from hyades.utils.share_memory_handler \
    import ShareBase
from hyades.config import Config

PROFILE = "profile"

# for raw data
RAW_DATA = "raw_data"
COORDINATOR = "coordinator"
ROUND = "round"

# for memo
MEMO = "memo"
CURRENT_ROUND = "current_round"


class Profiler(ShareBase):
    def __init__(self):
        super(Profiler, self).__init__(client_id=0)

    def get_profile_scheduling(self):
        pass

    def record_round_start_time(self, round_idx):
        self.set_a_shared_value(
            key=[PROFILE, RAW_DATA, COORDINATOR,
                 ROUND, round_idx, "start_time"],
            value=time.perf_counter()
        )

        self.set_a_shared_value(
            key=[PROFILE, MEMO, CURRENT_ROUND],
            value=round_idx
        )

    def record_round_end_time(self, round_idx):
        self.set_a_shared_value(
            key=[PROFILE, RAW_DATA, COORDINATOR,
                 ROUND, round_idx, "end_time"],
            value=time.perf_counter()
        )

    def log_print_raw_data(self):
        pass

