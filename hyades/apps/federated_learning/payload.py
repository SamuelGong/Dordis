import logging
import torch
import os, random
import numpy as np
from functools import reduce
from collections import OrderedDict


class Payload:
    def __init__(self):
        self.weight_info = {}
        self.num_chunks = None
        self.num_trainable_params = None
        self.chunk_size = None
        self.num_zeros_padded_at_the_last_chunk = 0

    def fix_randomness(self, seed):
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_data_dim(self):
        return self.num_trainable_params

    def set_chunk_size(self, chunk_size):  # should be a dict
        self.chunk_size = chunk_size
        self.num_chunks = len(chunk_size)
        actual_dim = reduce(lambda x, y: x + y,
                            list(chunk_size.values()))
        self.num_zeros_padded_at_the_last_chunk = \
            max(0, actual_dim - self.num_trainable_params)
        logging.info(f"# padded zeros at the last chunk: "
                     f"{self.num_zeros_padded_at_the_last_chunk}.")

    def extract_model_meta(self, model):
        weights = model.cpu().state_dict()  # weights is an OrderedDict
        self.weight_info['ordered_keys'] = list(weights.keys())
        self.weight_info['layer_shape'] = {}
        self.weight_info['trainable'] = []
        self.weight_info['non_trainable'] = []
        self.num_trainable_params = 0
        for k in weights.keys():
            shape = list(weights[k].size())
            if len(shape) > 0:
                self.weight_info['trainable'].append(k)
                self.weight_info['layer_shape'][k] = shape
                self.num_trainable_params += int(np.prod(shape))
            else:
                self.weight_info['non_trainable'].append(k)

        logging.info(f"FL weight_info: {self.weight_info}.")

    def weights_op(self, weights_1, weights_2, op="add"):
        res = {}
        non_trainable_list = self.weight_info['non_trainable']
        for k, v in weights_1.items():
            if k in non_trainable_list:
                res[k] = v
            else:
                if op == "add":
                    res[k] = v + weights_2[k]
                else:
                    res[k] = v - weights_2[k]

        return res

    def weights_to_chunks(self, weights, padding=True):
        weight_list = []
        non_trainable_list = self.weight_info['non_trainable']
        for k, v in weights.items():
            if k in non_trainable_list:
                continue
            weight_list += torch.flatten(v).tolist()

        cur = 0
        result = []
        for chunk_idx in sorted(self.chunk_size.keys()):
            chunk_size = self.chunk_size[chunk_idx]
            weight_chunk = weight_list[cur:cur+chunk_size]

            if chunk_idx < self.num_chunks - 1:
                assert chunk_size == len(weight_chunk)
            else:  # they may not equate due to the need for padding in DP
                if padding:
                    len_diff = max(0, chunk_size - len(weight_chunk))
                    weight_chunk += [0.0] * len_diff

            result.append(weight_chunk)
            cur += chunk_size
        return result

    def chunks_to_weights(self, chunks, non_trainable_dict, stripping=True):
        total_list = []
        for _chunk_idx, chunk in enumerate(chunks):
            if stripping:
                chunk = self.strip_possible_zeros(chunk, _chunk_idx)
            total_list += chunk

        cur = 0
        result = OrderedDict()
        for key in self.weight_info['ordered_keys']:
            if key in self.weight_info['trainable']:
                shape = self.weight_info['layer_shape'][key]
                size = np.prod(shape)
                result[key] = torch.Tensor(np.array(total_list[cur:cur+size])\
                    .reshape(shape))
                cur += size
            else:
                # if can be None. For example, if 1st round we have 8 physical clients
                # but 2nd round we have 9 physical clients,
                # then in the 2nd round the non_trainable_dict of 9-th client is None
                # TODO: to evaluate if it is meaningful to update non_trainable_dict in FL
                # when clients are not necessarily online all the time
                if non_trainable_dict is not None:
                    result[key] = non_trainable_dict[key]
        return result

    def extract_non_trainable_dict(self, weights):
        result = {}
        for key in weights.keys():
            if key in self.weight_info['non_trainable']:
                result[key] = weights[key]
        return result

    def strip_possible_zeros(self, data_chunk, chunk_idx):
        res = data_chunk
        if chunk_idx == self.num_chunks - 1 \
                and self.num_zeros_padded_at_the_last_chunk > 0:
            res = data_chunk[:-self.num_zeros_padded_at_the_last_chunk]
        return res
