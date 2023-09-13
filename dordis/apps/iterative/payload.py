import logging
import numpy as np
from functools import reduce
from dordis.config import Config
from dordis.primitives.differential_privacy.utils.misc \
    import clip_by_norm


class Payload:
    def __init__(self):
        self.num_chunks = None
        self.chunk_size = None
        self.data_dim = None
        self.num_zeros_padded_at_the_last_chunk = 0

    def get_data_dim(self):
        dim = Config().app.data.dim
        self.data_dim = dim
        return dim

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size
        self.num_chunks = len(chunk_size)
        actual_dim = reduce(lambda x, y: x + y,
                            list(chunk_size.values()))
        self.num_zeros_padded_at_the_last_chunk = \
                max(0, actual_dim - self.data_dim)
        logging.info(f"# padded zeros at the last chunk: "
                     f"{self.num_zeros_padded_at_the_last_chunk}.")

    def data_to_chunks(self, data):
        cur = 0
        result = []
        for chunk_idx in range(self.num_chunks):
            chunk_size = self.chunk_size[chunk_idx]
            data_chunk = data[cur:cur+chunk_size]
            if not isinstance(data_chunk, list):
                data_chunk = data_chunk.tolist()

            logging.info(f"chunk_size: {chunk_size}, "
                         f"len_data_chunk: {len(data_chunk)}.")
            if chunk_idx < self.num_chunks - 1:
                assert chunk_size == len(data_chunk)
            else:  # they may not equate due to the need for padding in DP
                len_diff = max(0, chunk_size - len(data_chunk))
                data_chunk += [0.0] * len_diff

            result.append(data_chunk)
            cur += chunk_size
        return result

    def generate_client_data(self, client_id, round_idx, chunk_idx=None):
        clients_data_params = Config().app.data
        source = clients_data_params.source

        if source in ["random"]:
            seed = clients_data_params.seed
            client_seed = seed * (client_id + 1) * (round_idx + 1)
            np.random.seed(client_seed)

            low, high = clients_data_params.range
            dim = clients_data_params.dim
            data = np.random.uniform(
                low=low, high=high, size=dim).astype(object)

            if hasattr(clients_data_params, "l2_clip_norm"):
                l2_clip_norm = clients_data_params.l2_clip_norm
                data = clip_by_norm(data, l2_clip_norm)
        else:
            raise ValueError('No such payload source: {}.'.format(source))

        if chunk_idx is None:  # return all
            return data
        else:
            cur = 0
            for _chunk_idx in sorted(self.chunk_size.keys()):
                chunk_size = self.chunk_size[_chunk_idx]
                if not _chunk_idx == chunk_idx:
                    cur += chunk_size
                else:
                    return data[cur:cur + chunk_size]

    def strip_possible_zeros(self, data_chunk, chunk_idx):
        res = data_chunk
        if chunk_idx == self.num_chunks - 1 \
                and self.num_zeros_padded_at_the_last_chunk > 0:
            res = data_chunk[:-self.num_zeros_padded_at_the_last_chunk]
        return res
