import copy
import logging
import pickle
import multiprocessing as mp
import time

from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.SecretSharing import Shamir
from dordis.primitives.secret_sharing import base
# N_JOBS = min(mp.cpu_count(), 16)
N_JOBS = mp.cpu_count()

from Crypto.Random import get_random_bytes as rng
from Crypto.Protocol.SecretSharing import _Element


def _get_factors_for_combine(idx_list, k):
    result = []
    for j in range(k):
        x_j = idx_list[j]

        numerator = _Element(1)
        denominator = _Element(1)

        for m in range(k):
            x_m = idx_list[m]
            if m != j:
                numerator *= x_m
                denominator *= x_j + x_m

        result.append(numerator * denominator.inverse())
    return result


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

    def create_shares(self, secret, t, n):
        padded_bytes = pad(
            data_to_pad=secret,
            block_size=self.block_size
        )
        # chunk_input_list = [
        #     (padded_bytes[i: i + self.block_size], t, n)
        #     for i in range(0, len(padded_bytes), 16)
        # ]

        # if mp.get_start_method(allow_none=True) != 'spawn':
        #     mp.set_start_method('spawn', force=True)
        # with mp.Pool(N_JOBS) as pool:
        #     pool_outputs = pool.starmap(self._create_shares,
        #                                 chunk_input_list)
        pool_outputs = []
        for i in range(0, len(padded_bytes), 16):
            pool_outputs.append(self._create_shares(padded_bytes[i: i + self.block_size], t, n))

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
        share_list = copy.deepcopy(share_list)
        for idx, share in enumerate(share_list):
            share_list[idx] = pickle.loads(share)

        chunk_num = len(share_list[0])
        chunk_shares_list = []
        for i in range(chunk_num):
            chunk_shares = []
            for j in range(len(share_list)):
                chunk_shares.append(share_list[j][i])
            chunk_shares_list.append(chunk_shares)

        # using first chunk's shares
        shares = chunk_shares_list[0]
        k = len(shares)
        idx_list = []
        for x in shares:
            idx = _Element(x[0])
            if any(y == idx for y in idx_list):
                raise ValueError("Duplicate share")
            idx_list.append(idx)

        return _get_factors_for_combine(idx_list=idx_list, k=k)

    @classmethod
    def _combine_shares(cls, shares, factors=None):
    # def _combine_shares(*shares):
        # the asterisk is needed if using mp.starmap (but it may slow down)
        k = len(shares)

        gf_shares = []
        for x in shares:
            idx = _Element(x[0])
            value = _Element(x[1])
            if any(y[0] == idx for y in gf_shares):
                raise ValueError("Duplicate share")
            # if ssss:
            #     value += idx ** k
            gf_shares.append((idx, value))

        if factors is None: # if it is not precomputed
            factors = _get_factors_for_combine(
                idx_list=[e[0] for e in gf_shares],
                k=k
            )

        result = _Element(0)
        for j in range(k):
            x_j, y_j = gf_shares[j]
            result += y_j * factors[j]
        return result.encode()

    def combine_shares(self, share_list, aux=None):
        if aux is not None:
            factors, = aux # aux should be a tuple
        else:
            factors = None
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
            pool_outputs.append(self._combine_shares(
                chunk_shares, factors=factors)
            )

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
