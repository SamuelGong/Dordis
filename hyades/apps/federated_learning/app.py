import time
import redis
import logging
import numpy as np
import torch
import multiprocessing as mp
from hyades.apps import base
from hyades.config import Config
from hyades.utils.debug import log_sketch
from hyades.apps.federated_learning.trainers \
    import registry as trainers_registry
from hyades.apps.federated_learning.datasources \
    import registry as datasources_registry
from hyades.apps.federated_learning.samplers \
    import registry as samplers_registry
from hyades.utils.share_memory_handler import redis_pool, \
    AGG_RES_USED_BY_SERVER, DATA_PREPARED, AGG_RES_USED_BY_CLIENT
from hyades.apps.federated_learning.payload import Payload
from hyades.utils.misc import get_chunks_idx

r = redis.Redis(connection_pool=redis_pool)
SERVER_SEED = 0


def _get_client_dataset_size(clients):
    result = {}
    for client_id in clients:
        temp_datasource = datasources_registry.get(
            client_id=client_id,
            quiet=True
        )
        result[client_id] = temp_datasource.num_train_examples()
    return result


def if_closest_pretrained_model_exists(existing_pretrained_models,
                                       target_num_rounds):
    _round_indices = []
    for pretrained_model_name in existing_pretrained_models:
        _round_idx = int(pretrained_model_name.split('-r')[-1])
        _round_indices.append(_round_idx)
    if len(_round_indices) == 0:
        return None

    _round_indices = sorted(_round_indices)
    _i = None
    for _j, _round_idx in enumerate(_round_indices):
        if _round_idx > target_num_rounds:
            _i = _j - 1

    # if there is a cloest pretrained model that can be reused
    if _i is None or not _i == -1:  # not _j == 0
        if _i is None:
            _i = -1
        closest_pretrained_model_round_idx = _round_indices[_i]
        return closest_pretrained_model_round_idx
    else:
        return None


class AppServer(base.AppServer, Payload):
    def __init__(self, client_id=0):
        Payload.__init__(self)
        base.AppServer.__init__(self, client_id=0)

        self.aggregate_delta = Config().app.aggregate_delta \
            if hasattr(Config().app, "aggregate_delta") \
            else True
        self.fix_randomness(seed=SERVER_SEED)

        self.trainer = trainers_registry.get()
        self.trainer.set_client_id(client_id=0)
        self.datasource = datasources_registry.get(
            client_id=0)
        self.testset = self.datasource.get_test_set()

        self.extract_model_meta(model=self.trainer.model)

        self.model_save_interval = 0
        if hasattr(Config().results, "model_save_interval"):
            self.model_save_interval = Config().results.model_save_interval

        self.to_start_from_pretrained_model = False
        if hasattr(Config().app.trainer, "start_from_pretrained_model"):
            self.to_start_from_pretrained_model \
                = Config().app.trainer.start_from_pretrained_model

    def if_start_from_pretrained_model(self):
        return self.to_start_from_pretrained_model

    def if_closest_pretrained_model_exists(self):
        existing_pretrained_models = self.trainer.list_saved_models()
        target_num_rounds = Config().app.repeat
        closest_pretrained_model_round_idx \
            = if_closest_pretrained_model_exists(
            existing_pretrained_models,
            target_num_rounds
        )
        return closest_pretrained_model_round_idx

    def get_client_dataset_size_dict(self, clients, log_prefix_str):
        logging.info(f"{log_prefix_str} "
                     f"Obtaining clients' dataset sizes...")

        # multiprocessing
        N_JOBS = 16  # TODO: avoid hard-coding

        pool_inputs = []
        worker_function = _get_client_dataset_size
        for begin, end in get_chunks_idx(
                l=len(clients),
                n=N_JOBS
        ):
            pool_inputs.append(
                (clients[begin:end],)
            )

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        with mp.Pool(N_JOBS) as pool:
            pool_outputs = pool.starmap(worker_function, pool_inputs)

        result = {}
        for d in pool_outputs:
            result.update(d)

        logging.info(f"{log_prefix_str} "
                     f"Clients' dataset obtained: {result}.")
        return result

    def debug_and_test(self, agg_res, involved_clients, log_prefix_str,
                       round_idx, chunk_idx):
        flatten_weights = []
        for _chunk_idx, chunk in enumerate(agg_res['chunks']):
            flatten_weights += chunk

        log_sketch(
            data=flatten_weights,
            log_prefix_str=log_prefix_str
        )

        if self.debug \
                and hasattr(Config().app.debug.server, "test") \
                and Config().app.debug.server.test is True:
            # self.trainer.load_weights(agg_res["weights"])
            accuracy = self.trainer.server_test(self.testset)
            logging.info(f"{log_prefix_str} [Debug]"
                         f" Testing accuracy: {round(accuracy, 4)}.")

    def use_output(self, args):
        round_idx, chunk_idx, log_prefix_str, \
            agg_res, involved_clients = args

        # averaging model updates
        agg_res = self.strip_possible_zeros(agg_res, chunk_idx)  # related to DP
        chunk = np.array(agg_res) / len(involved_clients)
        logging.info(f"{log_prefix_str} Model updates aggregated.")
        logging.info(f"[Debug] Chunk after aggregation, "
                     f"first 6: {[round(e, 4) for e in chunk[:6]]}, "
                     f"last 6: {[round(e, 4) for e in chunk[-6:]]}.")

        first_global_round_done = self.get_a_shared_value("first_global_round_done")
        if first_global_round_done is None:
            # Consider possible pretrained model
            if self.to_start_from_pretrained_model:
                existing_pretrained_models = self.trainer.list_saved_models()
                target_num_rounds = Config().app.repeat

                closest_pretrained_model_round_idx = if_closest_pretrained_model_exists(
                    existing_pretrained_models,
                    target_num_rounds
                )
                if closest_pretrained_model_round_idx is not None:
                    closest_pretrained_model_filename = Config().app.trainer.model_name
                    closest_pretrained_model_filename += f"-r{closest_pretrained_model_round_idx}"

                    self.trainer.load_model(closest_pretrained_model_filename)
                    logging.info(f"Using pretrained model named "
                                 f"{closest_pretrained_model_filename}.")

            self.set_a_shared_value(
                key="first_global_round_done",
                value=0  # placeholder
            )

        # otherwise the trainer's model is just the last-round one, or
        # a randomly initialzied one when this is the first global round)
        base_weights = self.trainer.extract_weights()
        if first_global_round_done is None and chunk_idx == 0:
            _chunks = self.weights_to_chunks(base_weights, padding=False)
            logging.info(f"[Debug] Initialized state, "
                         f"first 6: {[round(e, 4) for e in _chunks[0][:6]]}, "
                         f"last 6: {[round(e, 4) for e in _chunks[0][-6:]]}.")

        if self.aggregate_delta:
            base_weights_chunks = self.weights_to_chunks(
                base_weights,
                padding=False
            )
            base_weight_chunk = base_weights_chunks[chunk_idx]
            chunk += base_weight_chunk  # now it becomes weight chunks
        self.set_a_shared_value(
            key=["output", round_idx, chunk_idx],
            value=chunk,
        )

        # see if it is the last chunks
        keys = self.prefix_to_dict(prefix=["output", round_idx]).keys()
        if len(keys) == self.num_chunks:
            weight_chunks = []
            for chunk_idx in range(self.num_chunks):
                data = self.get_a_shared_value(
                    key=['output', round_idx, chunk_idx]
                )
                self.delete_a_shared_value(
                    key=['output', round_idx, chunk_idx]
                )
                weight_chunks.append(data.tolist())

            non_trainable_dict = self.extract_non_trainable_dict(base_weights)
            weights = self.chunks_to_weights(
                chunks=weight_chunks,
                non_trainable_dict=non_trainable_dict,
                stripping=False
            )

            # update the global model unconditionally
            # (it is not for dispatching but model debug and saving)
            self.trainer.load_weights(weights)

            self.debug_and_test(
                agg_res={'chunks': weight_chunks, 'weights': weights},
                involved_clients=involved_clients,
                log_prefix_str=log_prefix_str,
                round_idx=round_idx,
                chunk_idx=chunk_idx
            )
            # # need to memorize the server's global model
            # if self.aggregate_delta:
            #     self.trainer.load_weights(weights)

            logical_round_idx = round_idx + 1
            if self.model_save_interval > 0 and logical_round_idx \
                    % self.model_save_interval == 0:
                model_name = Config().app.trainer.model_name
                model_name += f"-r{round_idx}"
                self.trainer.save_model(filename=model_name)

        self._publish_a_value(
            channel=[AGG_RES_USED_BY_SERVER, round_idx, chunk_idx],
            message=chunk,
            mode="large"
        )


class AppClient(base.AppClient, Payload):
    def __init__(self, client_id):
        Payload.__init__(self)
        base.AppClient.__init__(self, client_id)

        self.aggregate_delta = Config().app.aggregate_delta \
            if hasattr(Config().app, "aggregate_delta") \
            else True

        # for the first selected clients, they need to begin with the
        # same place as the server does
        original_seed = Config().app.data.random_seed
        self.fix_randomness(SERVER_SEED) # this is what the server uses
        self.trainer = trainers_registry.get()

        # configuring self randomness
        self.fix_randomness(seed=original_seed * client_id)

        self.trainer.set_client_id(client_id=client_id)
        self.datasource = datasources_registry.get(
            client_id=self.client_id)
        self.trainset = self.datasource.get_train_set()
        self.sampler = samplers_registry.get(self.datasource, self.client_id)

        self.extract_model_meta(model=self.trainer.model)

        self.to_start_from_pretrained_model = False
        if hasattr(Config().app.trainer, "start_from_pretrained_model"):
            self.to_start_from_pretrained_model \
                = Config().app.trainer.start_from_pretrained_model

    def prepare_data(self, args):
        round_idx, chunk_idx, log_prefix_str, logical_client_id = args

        if hasattr(Config().agg, "pipeline") \
                and not Config().agg.pipeline.type == "even":
            raise NotImplementedError # TODO
        else:  # evenly chunking (including the trivial case)
            if chunk_idx == 0:
                # only in resource_saving mode
                # that we might encounter a varying logical_client_id
                if hasattr(Config().clients, "resource_saving") \
                        and Config().clients.resource_saving:
                    self.datasource = datasources_registry.get(
                        client_id=logical_client_id)
                    self.trainset = self.datasource.get_train_set()
                    self.sampler = samplers_registry.get(
                        self.datasource, logical_client_id
                    )

                base_weights = self.get_a_shared_value(
                    key=["previous_model", round_idx - 1]
                )
                if base_weights is not None:
                    self.delete_a_shared_value(
                        key=["previous_model", round_idx - 1]
                    )
                    self.trainer.load_weights(base_weights)
                else:  # should be the first global round

                    # Loading pretrained models
                    if self.to_start_from_pretrained_model:
                        existing_pretrained_models = self.trainer.list_saved_models()
                        target_num_rounds = Config().app.repeat

                        closest_pretrained_model_round_idx = if_closest_pretrained_model_exists(
                            existing_pretrained_models,
                            target_num_rounds
                        )
                        if closest_pretrained_model_round_idx is not None:
                            closest_pretrained_model_filename = Config().app.trainer.model_name
                            closest_pretrained_model_filename += f"-r{closest_pretrained_model_round_idx}"

                            self.trainer.load_model(closest_pretrained_model_filename)
                            logging.info(f"Using pretrained model named "
                                         f"{closest_pretrained_model_filename}.")

                    # otherwise the trainer's model is just randomly initialized
                    # Seeing the intial model state
                    chunks = self.weights_to_chunks(base_weights, padding=False)
                    logging.info(f"[Debug] Initialized state,  "
                                 f"first 6: {[round(e, 6) for e in chunks[0][:6]]}, "
                                 f"last 6: {[round(e, 6) for e in chunks[0][-6:]]}.")
                try:
                    logging.info("%s FL training started.", log_prefix_str)
                    self.trainer.train(self.trainset, self.sampler)
                    logging.info("%s FL training ended.", log_prefix_str)
                except ValueError as e:
                    logging.info("%s FL training error: %s.", log_prefix_str, e)
                    raise e

                weights = self.trainer.extract_weights()
                non_trainable_dict = self.extract_non_trainable_dict(weights)

                if self.aggregate_delta: # send model update
                    weights_diff = self.weights_op(
                        weights, base_weights, op="subtract")
                    chunks = self.weights_to_chunks(weights_diff)
                else: # send model
                    chunks = self.weights_to_chunks(weights)

                for _chunk_idx in range(1, self.num_chunks):
                    self.set_a_shared_value(
                        key=['model', _chunk_idx],
                        value=chunks[_chunk_idx]
                    )

                chunk = chunks[chunk_idx]
                self.set_a_shared_value(
                    key=["non_trainable", round_idx],
                    value=non_trainable_dict
                )

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

            else:  # has already been calculated
                chunk = self.get_a_shared_value(key=['model', chunk_idx])
                while chunk is None:
                    time.sleep(0.1)
                    chunk = self.get_a_shared_value(key=['model', chunk_idx])
                self.delete_a_shared_value(key=['model', chunk_idx])

        log_sketch(
            data=self.strip_possible_zeros(chunk, chunk_idx),
            log_prefix_str=log_prefix_str,
            mode="client"
        )

        if chunk_idx == self.num_chunks - 1:
            utility = self.get_a_shared_value(key=['model_utility'])
            self._publish_a_value(
                channel=[DATA_PREPARED, round_idx, chunk_idx],
                message={
                    "chunk": chunk,
                    "utility": utility
                },
                mode="large"
            )
        else:
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
            value={
                "agg_res": agg_res,
                "involved_clients": involved_clients
            }
        )

        # TODO: we have not accounted for the case when some chunks are missed
        keys = self.prefix_to_dict(prefix=["output", round_idx]).keys()
        if len(keys) == self.num_chunks:
            chunks = []
            for chunk_idx in range(self.num_chunks):
                chunk_output = self.get_a_shared_value(
                    key=['output', round_idx, chunk_idx]
                )
                self.delete_a_shared_value(
                    key=['output', round_idx, chunk_idx]
                )

                logging.info(f'[Debug] {Config().agg.type}: need tolist? '
                             f'{isinstance(chunk_output["agg_res"], np.ndarray)}.')
                if isinstance(chunk_output["agg_res"], np.ndarray):
                    chunks.append(chunk_output["agg_res"].tolist())
                else:  # already a list
                    chunks.append(chunk_output["agg_res"])

            non_trainable_dict = self.get_a_shared_value(
                key=["non_trainable", round_idx]
            )
            self.delete_a_shared_value(key=["non_trainable", round_idx])
            # whether self.aggregate_delta is true or not
            # the received data must be a complete model
            weights = self.chunks_to_weights(
                chunks=chunks,
                non_trainable_dict=non_trainable_dict,
                stripping=False
            )
            # each physical client receives a global model and stores it as "previous_model" for the next round
            # the client who stores it and the one who uses it are consistent
            self.set_a_shared_value(
                key=["previous_model", round_idx],
                value=weights
            )
            for chunk_idx in range(self.num_chunks):
                self._publish_a_value(
                    channel=[AGG_RES_USED_BY_CLIENT, round_idx, chunk_idx],
                    message=1  # placeholder
                )
