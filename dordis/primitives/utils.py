import os
import pickle


def rand_bytes(num=32):
    return os.urandom(num)


# def secagg_concatenate(src_client_id, dst_client_id,
#                        b_share, s_sk_share):
#     return pickle.dumps([src_client_id, dst_client_id,
#                          b_share, s_sk_share])

def secagg_concatenate(*args):
    return pickle.dumps(list(args))

def secagg_separate(concatenated_message):
    return tuple(pickle.loads(concatenated_message))
