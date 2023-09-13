import time
import redis
import logging
import numpy as np
from dordis.apps import base
from dordis.utils.debug import log_sketch, validate_result
from dordis.utils.misc import plaintext_aggregate
from dordis.apps.iterative.payload import Payload
from dordis.config import Config
from dordis.utils.share_memory_handler import redis_pool, \
    AGG_RES_USED_BY_SERVER, DATA_PREPARED, AGG_RES_USED_BY_CLIENT

r = redis.Redis(connection_pool=redis_pool)


class AppServer(base.AppServer, Payload):
    def __init__(self, client_id=0):
        Payload.__init__(self)
        base.AppServer.__init__(self, client_id=0)

    def debug_and_validate(self, agg_res, involved_clients, log_prefix_str,
                           round_idx, chunk_idx):
        # if hasattr(Config().app, "mean") and Config().app.mean:
        #     agg_res = (np.array(agg_res)
        #                        / len(involved_clients)).tolist()

        # for the last chunk of agg_res
        # it may be padded with zeros
        log_sketch(
            data=agg_res,
            log_prefix_str=log_prefix_str
        )
        if self.debug:
            if hasattr(Config().app.debug.server, "test") \
                and Config().app.debug.server.test is True:
                test_clients_data = []
                for client_id in involved_clients:
                    client_data = self.generate_client_data(
                        client_id=client_id,
                        round_idx=round_idx,
                        chunk_idx=chunk_idx
                    )
                    test_clients_data.append(client_data)
                test_agg_res = plaintext_aggregate(
                    data_list=test_clients_data
                )

                # if hasattr(Config().app, "mean") and Config().app.mean:
                #     test_agg_res = (np.array(test_agg_res)
                #                             / len(involved_clients)).tolist()
                log_sketch(
                    data=test_agg_res,
                    log_prefix_str=log_prefix_str,
                    validate=True
                )

                validate_result(
                    res=agg_res,
                    test_res=test_agg_res,
                    prefix_string=log_prefix_str,
                    involved_clients=involved_clients
                )

    def use_output(self, args):
        round_idx, chunk_idx, log_prefix_str, \
            agg_res, involved_clients = args
        agg_res = self.strip_possible_zeros(agg_res, chunk_idx)

        self.debug_and_validate(
            agg_res=agg_res,
            involved_clients=involved_clients,
            log_prefix_str=log_prefix_str,
            round_idx=round_idx,
            chunk_idx=chunk_idx,
        )

        self._publish_a_value(
            channel=[AGG_RES_USED_BY_SERVER, round_idx, chunk_idx],
            message=agg_res,
            mode="large"
        )


class AppClient(base.AppClient, Payload):
    def __init__(self, client_id):
        Payload.__init__(self)
        base.AppClient.__init__(self, client_id)

    def prepare_data(self, args):
        round_idx, chunk_idx, log_prefix_str, logical_client_id = args

        # logging.info(f"round_idx: {round_idx}, "
        #              f"chunk_idx: {chunk_idx}, "
        #              f"logical_client_id: {logical_client_id}, "
        #              f"num_chunks: {self.num_chunks}.")

        if chunk_idx == 0:
            start_time = time.perf_counter()
            data = self.generate_client_data(
                client_id=logical_client_id,
                round_idx=round_idx
            )
            logging.info(f"Data generated for "
                         f"logical client #{logical_client_id}.")
            log_sketch(
                data=data,
                log_prefix_str=log_prefix_str,
                mode="client"
            )
            self.mocking_preparation_time(
                start_time=start_time,
                num_elements=len(data),
                log_prefix_str=log_prefix_str
            )

            chunks = self.data_to_chunks(data)
            for _chunk_idx in range(1, self.num_chunks):
                self.set_a_shared_value(
                    key=['generated_data', _chunk_idx],
                    value=chunks[_chunk_idx]
                )
            chunk = chunks[chunk_idx]

            # For DP protocols to clip properly
            protocol_type = Config().agg.type
            if protocol_type == "dp" \
                    or protocol_type == "dp_plus_secagg":
                norm_square = 0.0
                for _chunk_idx in range(self.num_chunks):
                    norm_square += sum(np.square(chunks[_chunk_idx]))
                norm = np.sqrt(norm_square)
                self.set_a_shared_value(
                    key=["full_data_norm", round_idx],
                    value=norm
                )
        else:
            chunk = self.get_a_shared_value(key=['generated_data', chunk_idx])
            while chunk is None:
                time.sleep(0.1)
                chunk = self.get_a_shared_value(key=['generated_data', chunk_idx])
            self.delete_a_shared_value(key=['generated_data', chunk_idx])

        self._publish_a_value(
            channel=[DATA_PREPARED, round_idx, chunk_idx],
            message=chunk,
            mode="large"
        )

    def use_output(self, args):
        round_idx, chunk_idx, log_prefix_str, \
            agg_res, involved_clients, logical_client_id = args
        self.set_a_shared_value(
            key=["output", round_idx, chunk_idx],
            value=1
        )

        # TODO: we have not accounted for the case when some chunks are missed
        keys = self.prefix_to_dict(prefix=["output", round_idx]).keys()
        if len(keys) == self.num_chunks:
            for chunk_idx in range(self.num_chunks):
                self._publish_a_value(
                    channel=[AGG_RES_USED_BY_CLIENT, round_idx, chunk_idx],
                    message=1  # placeholder
                )
