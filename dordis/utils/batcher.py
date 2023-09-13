import multiprocessing as mp
from dordis.utils.misc \
    import get_chunks_idx, get_chunks_idx_with_mod

# N_JOBS = min(mp.cpu_count(), 4)
# N_JOBS = mp.cpu_count()
N_JOBS = 1


def _batch(array, batch_size, bits_per_element):
    res = []
    begin = 0
    while begin < len(array):
        end = begin + batch_size
        if end > len(array):
            end = len(array)

        temp = 0
        for element in array[begin:end]:
            temp <<= bits_per_element
            temp += element
        res.append(temp)

        begin += batch_size
    return res


def _unbatch(array, bits_per_element,
             batch_size, num_unbatched_elements):
    res = []
    mod = (1 << bits_per_element) - 1
    for batched_element in array:
        reverse_temp = []
        n = min(batch_size, num_unbatched_elements)
        num_unbatched_elements -= n
        while n:
            reverse_temp.append(batched_element & mod)
            batched_element >>= bits_per_element
            n -= 1
        reverse_temp.reverse()
        res += reverse_temp

    return res


def batch(flatten_array, bits_per_element, batch_size,
          original_length=None):
    pool_inputs = []

    if N_JOBS == 1:  # avoid multiprocessing for run time stability
        if original_length is not None:  # unbatch
            res = _unbatch(
                array=flatten_array,
                bits_per_element=bits_per_element,
                batch_size=batch_size,
                num_unbatched_elements=original_length
            )
        else:
            res = _batch(
                array=flatten_array,
                bits_per_element=bits_per_element,
                batch_size=batch_size
            )
    else:  # actually need multiprocessing
        if original_length is not None:  # unbatch
            worker_function = _unbatch
            for begin, end in get_chunks_idx(
                    l=len(flatten_array),
                    n=N_JOBS
            ):
                num_unbatched_elements = min((end - begin) * batch_size,
                                             original_length)
                original_length -= num_unbatched_elements
                pool_inputs.append(
                    [flatten_array[begin:end],
                     bits_per_element,
                     batch_size,
                     num_unbatched_elements]
                )
        else:  # batch
            mod = batch_size
            worker_function = _batch
            for begin, end in get_chunks_idx_with_mod(
                    l=len(flatten_array),
                    n=N_JOBS,
                    mod=mod
            ):
                pool_inputs.append(
                    [flatten_array[begin:end],
                     batch_size,
                     bits_per_element]
                )

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        with mp.Pool(N_JOBS) as pool:
            pool_outputs = pool.starmap(worker_function, pool_inputs)

        res = []
        for arr in pool_outputs:
            res += arr
    return res
