import numpy as np

dropout_rate_list = [0, 0.1, 0.2, 0.3]
num_sampled_client_list = [100, 200, 300]
model_size_list = [5000000, 50000000, 500000000]
dropout_tolerance = 0.5
secret_sharing_threshold = 0.5

bits_per_bytes = 8
bytes_per_element = 20 / bits_per_bytes
print(bytes_per_element)
bytes_per_seed = 32
bytes_per_secret = 16
bytes_per_ciphertext = 120  # due to encryption
bandwidth_in_bytes_per_second = 10 * 1e6 / bits_per_bytes
bytes_per_MB = 1024 * 1024
bandwidth_in_MB_per_second = bandwidth_in_bytes_per_second / bytes_per_MB

print("For a client:")
for dropout_rate in dropout_rate_list:
    print(f"Dropout {dropout_rate}")
    for num_sampled_client in num_sampled_client_list:
        num_dropped_clients = int(np.floor(num_sampled_client * dropout_rate))
        num_surviving_clients = num_sampled_client - num_dropped_clients

        print(f"\tSampled clients: {num_sampled_client}")
        for model_size in model_size_list:
            print(f"\t\tModel size: {model_size}")
            strawman_additional_network_in_MB = model_size * bytes_per_element / bytes_per_MB
            strawman_additional_network_time = strawman_additional_network_in_MB \
                                               / bandwidth_in_MB_per_second
            print(f"\t\t\tStrawman: network {round(strawman_additional_network_in_MB, 2)}MB, "
                  f"time: {round(strawman_additional_network_time, 2)}s.")

            num_tolerated_clients = int(np.floor(num_sampled_client * dropout_tolerance))
            # secret sharing before aggregation
            xnoise_additional_network_in_MB \
                = (num_tolerated_clients - 1) * num_tolerated_clients \
                  * bytes_per_ciphertext * 2 / bytes_per_MB
            # send and receive

            # best case (only sending my seeds whose number is T - D)
            # sending seeds after aggregation
            xnoise_additional_network_in_MB \
                += (num_tolerated_clients - num_dropped_clients) * bytes_per_seed / bytes_per_MB
            xnoise_additional_network_time = xnoise_additional_network_in_MB \
                                             / bandwidth_in_MB_per_second
            print(f"\t\t\tXNoise (best): network {round(xnoise_additional_network_in_MB, 2)}MB, "
                  f"time: {round(xnoise_additional_network_time, 2)}s.")

            # worst case
            max_allowed_num_dropped_clients = int(np.ceil((1 - secret_sharing_threshold) * num_sampled_client))
            num_dropped_clients_additional = max_allowed_num_dropped_clients - num_dropped_clients
            # send shares for each of the additionally dropped clients
            xnoise_additional_network_in_MB \
                += num_dropped_clients_additional * bytes_per_secret \
                   * (num_tolerated_clients - num_dropped_clients) / bytes_per_MB
            xnoise_additional_network_time = xnoise_additional_network_in_MB \
                                             / bandwidth_in_MB_per_second
            print(f"\t\t\tXNoise (worst): network {round(xnoise_additional_network_in_MB, 2)}MB, "
                  f"time: {round(xnoise_additional_network_time, 2)}s.")
