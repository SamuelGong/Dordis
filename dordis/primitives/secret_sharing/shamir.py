import logging
import pickle
import multiprocessing as mp
import time

from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.SecretSharing import Shamir
from dordis.primitives.secret_sharing import base
# N_JOBS = min(mp.cpu_count(), 16)
N_JOBS = mp.cpu_count()


class Handler(base.Handler):
    def __init__(self):
        super().__init__()
        self.block_size = 16

    @staticmethod
    def _create_shares(chunk, t, n):
        return Shamir.split(
            k=t,
            n=n,
            secret=chunk
        )

    @staticmethod
    def _combine_shares(shares):
    # def _combine_shares(*shares):
        # the asterisk is needed if using mp.starmap (but it may slow down)
        return Shamir.combine(
            shares=shares
        )

    def create_shares(self, secret, t, n):
        padded_bytes = pad(
            data_to_pad=secret,
            block_size=self.block_size
        )
        chunk_input_list = [
            (padded_bytes[i: i + self.block_size], t, n)
            for i in range(0, len(padded_bytes), 16)
        ]

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        with mp.Pool(N_JOBS) as pool:
            pool_outputs = pool.starmap(self._create_shares,
                                        chunk_input_list)

        share_list = []
        for _ in range(n):
            share_list.append([])
        for chunk_shares in pool_outputs:
            for idx, share in chunk_shares:
                # idx start with 1
                share_list[idx - 1].append((idx, share))
        for idx, shares in enumerate(share_list):
            share_list[idx] = pickle.dumps(shares)

        return share_list

    @classmethod
    def get_factors_for_combine(cls, share_list):
        return None

    def combine_shares(self, share_list, aux=None):
        # aux is not use in this case
        for idx, share in enumerate(share_list):
            share_list[idx] = pickle.loads(share)

        chunk_num = len(share_list[0])
        chunk_shares_list = []
        for i in range(chunk_num):
            chunk_shares = []
            for j in range(len(share_list)):
                chunk_shares.append(share_list[j][i])
            chunk_shares_list.append(chunk_shares)

        # using mp here can slow down the first iteration
        # if mp.get_start_method(allow_none=True) != 'spawn':
        #     mp.set_start_method('spawn', force=True)
        # with mp.Pool(N_JOBS) as pool:
        #     pool_outputs = pool.starmap(self._combine_shares,
        #                                 chunk_shares_list)
        pool_outputs = []
        for chunk_shares in chunk_shares_list:
            pool_outputs.append(self._combine_shares(chunk_shares))

        secret_padded = bytearray(0)
        for chunk in pool_outputs:
            secret_padded += chunk
        try:
            secret = unpad(
                padded_data=secret_padded,
                block_size=self.block_size
            )
        except ValueError as e:
            logging.info(f"Unable to recover the secret ({e}).")
            return None
        else:
            return bytes(secret)
