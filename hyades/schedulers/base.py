import json
import logging
import time
import redis
import pickle
import numpy as np
from functools import reduce
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool
from hyades.utils.share_memory_handler \
    import ShareBase, SCHEDULE, redis_pool, \
    SAMPLED_CLIENTS, BREAK_SCHEDULER_LOOP
from hyades.config import Config

# N_CORES = min(cpu_count(), 16)
N_CORES = cpu_count()
r = redis.Redis(connection_pool=redis_pool)


class PhaseNodePerRound:
    def __init__(self, id, params, num_chunks):
        self.id = id
        self.worker = params["worker"]
        self.predecessor_ids = params["predecessor_ids"]
        self.successor_id = params["successor_id"]
        self.stage = params["stage"]
        self.chunks_completed = [False] * num_chunks

        # chunk_idx (int): phase_idx_list (list of int) (waitee)
        # dependant chunk (int): {
        #     'chunk_idx': chunk_holding_the_resource (int),
        #     'phase_idx_list': phases_holding_the_resource (int)
        # }
        self.resource_dependency = params["resource_dependency"]

        # chunk_being_waited (int): {
        #     waiting_chunk (int): list_of_waiting_phases (list)
        # }
        self.waited_by = {}


class Scheduler(ShareBase):
    def __init__(self, hyades_instance, log_prefix_str):
        super(Scheduler, self).__init__(client_id=0)
        self.hyades_server = hyades_instance
        self.log_prefix_str = log_prefix_str
        self.final_node_id = None
        self.round_dict = {}
        self.graph_dict = None
        self.round = 0
        self.num_chunks = None
        self.leaf_phases = []
        self.total_clients = Config().clients.total_clients

    def set_num_chunks(self, num_chunks):
        self.num_chunks = num_chunks
        self.find_resource_dependencies()

        # for debug use
        nice_prompt = json.dumps(
            self.graph_dict, sort_keys=False, indent=4)
        logging.info(f"graph_dict: {nice_prompt}.")

    def register_graph(self, graph_dict):
        """
        sample item in graph_dict (the input):
        {id: {
            worker: [method]
            predecessor_ids: [list of id]
            is_final: [bool, optional]
        }}
        """
        for phase_idx, d in graph_dict.items():
            if "is_final" in d and d["is_final"]:
                self.final_node_id = phase_idx
            if not d["predecessor_ids"]:
                self.leaf_phases.append(phase_idx)
            graph_dict[phase_idx]["successor_id"] = None

        self.graph_dict = graph_dict
        self.set_successor_id()

    def set_successor_id(self):
        for phase_idx, phase_dict in self.graph_dict.items():
            predecessors = phase_dict['predecessor_ids']
            for p in predecessors:
                self.graph_dict[p]["successor_id"] = phase_idx

    def find_predecessors_with_same_stage(self, phase_idx,
                                          target_stage, diff=False):
        current_phase = self.graph_dict[phase_idx]
        if not current_phase["predecessor_ids"]:
            return []

        res = []
        for predecessor_idx in current_phase["predecessor_ids"]:
            predecessor_dict = self.graph_dict[predecessor_idx]
            predecessor_stage = predecessor_dict["stage"]

            if predecessor_stage == target_stage:
                if diff:  # need to find same stage that is not adjacent
                    res += [predecessor_idx]
                else:
                    res += self.find_predecessors_with_same_stage(
                        phase_idx=predecessor_idx,
                        target_stage=target_stage
                    )
            else:
                res += self.find_predecessors_with_same_stage(
                    phase_idx=predecessor_idx,
                    target_stage=target_stage,
                    diff=True
                )
        return res

    def find_resource_dependencies(self):
        for phase_idx, d in self.graph_dict.items():
            self.graph_dict[phase_idx]["resource_dependency"] = {}
            for chunk_idx in range(self.num_chunks):
                self_stage = d["stage"]
                if chunk_idx == 0:
                    predecessors_with_same_stage \
                        = self.find_predecessors_with_same_stage(
                        phase_idx=phase_idx,
                        target_stage=self_stage
                    )
                    self.graph_dict[phase_idx]["resource_dependency"]\
                        [chunk_idx] = {
                        'chunk_idx': self.num_chunks - 1,
                        'phase_idx_list': predecessors_with_same_stage
                    }
                else:
                    _phase_idx = phase_idx
                    _phase_dict = d
                    while _phase_dict["successor_id"] is not None:
                        successor_idx = _phase_dict["successor_id"]
                        if successor_idx == self.final_node_id:
                            break

                        successor_stage = self.graph_dict[
                            successor_idx]["stage"]
                        if not successor_stage == self_stage:
                            break
                        _phase_idx = successor_idx
                        _phase_dict = self.graph_dict[_phase_idx]

                    self.graph_dict[phase_idx]["resource_dependency"]\
                        [chunk_idx] = {
                        'chunk_idx': chunk_idx - 1,
                        'phase_idx_list': [_phase_idx]
                    }

    def if_resource_available(self, round_idx, chunk_idx,
                              phase_idx, set_waited=False):
        resource_available = True
        phase = self.round_dict[round_idx][phase_idx]

        if phase.resource_dependency \
                and chunk_idx in phase.resource_dependency:
            d = phase.resource_dependency[chunk_idx]
            chunk_idx_to_wait = d["chunk_idx"]
            phase_idx_list_to_wait = d["phase_idx_list"]  # is a list

            for phase_idx_to_wait in phase_idx_list_to_wait:
                phase_to_wait = self.round_dict[round_idx][phase_idx_to_wait]
                if not phase_to_wait.chunks_completed[chunk_idx_to_wait]:
                    resource_available = False
                    if set_waited:
                        if chunk_idx_to_wait not in phase_to_wait.waited_by:
                            phase_to_wait.waited_by[chunk_idx_to_wait] = {}

                        if chunk_idx in phase_to_wait\
                                .waited_by[chunk_idx_to_wait]:
                            self.round_dict[round_idx][phase_idx_to_wait]\
                                .waited_by[chunk_idx_to_wait][chunk_idx]\
                                .append(phase_idx)
                        else:
                            self.round_dict[round_idx][phase_idx_to_wait] \
                                .waited_by[chunk_idx_to_wait][chunk_idx] \
                                = [phase_idx]
                    else:
                        break

        return resource_available

    def if_all_siblings_done(self, round_idx, chunk_idx, phase_idx):
        all_siblings_done = True
        phase = self.round_dict[round_idx][phase_idx]

        for sibling_phase in phase.predecessor_ids:
            sibling = self.round_dict[round_idx][sibling_phase]
            if not sibling.chunks_completed[chunk_idx]:
                all_siblings_done = False
                break

        return all_siblings_done

    def new_a_round(self, round_idx, num_chunks):
        if round_idx == 0:  # sampling clients
            available_clients = self.hyades_server.client_manager.get_available_clients()
            logging.info(f"{self.log_prefix_str} [Round {round_idx}] "
                         f"Available clients: {available_clients}.")
            self.hyades_server.client_sampler.sample(
                candidates=available_clients,  # TODO: to aware of dropout
                round_idx=round_idx,
                log_prefix_str=self.log_prefix_str
            )

        sampled_clients = self.get_a_shared_value(
            key=[SAMPLED_CLIENTS, round_idx]
        )
        if len(sampled_clients) > 0:  # only start if sufficient clients are sampled
            logging.info(f"{self.log_prefix_str} [Round {round_idx}] "
                         f"Sampled clients ({len(sampled_clients)}): {sampled_clients}.")

            # start the application
            node_dict = {}
            for phase_idx, d in self.graph_dict.items():
                new_node = PhaseNodePerRound(phase_idx, d, num_chunks)
                node_dict[phase_idx] = new_node
            self.round_dict[round_idx] = node_dict

            for leaf_phase_id in self.leaf_phases:
                # leaf_node = self.round_dict[round_idx][leaf_phase_id]
                for chunk_idx in range(self.num_chunks):
                    self.hyades_server.protocol.execute_a_task(
                        task_info=(round_idx, chunk_idx, leaf_phase_id)
                    )
        else:
            assert 0  # should not reach here. If no available clients, should exist earlier

    def schedule(self):
        logging.info("%s Starting round %d.",
                     self.log_prefix_str, self.round)
        # bootstrap
        self.new_a_round(
            round_idx=self.round,
            num_chunks=self.num_chunks
        )

        sub, ch_dict = self.batch_subscribe_channels(
            d={
                SCHEDULE: False,
                BREAK_SCHEDULER_LOOP: False
            }
        )
        schedule_ch = ch_dict[SCHEDULE]
        break_ch = ch_dict[BREAK_SCHEDULER_LOOP]

        for message in sub.listen():
            raw_data = message['data']
            if not isinstance(raw_data, bytes):
                continue
            channel = message['channel'].decode()
            channel = self.strip_self_channel_prefix(channel)

            if channel == break_ch:  # e.g., break by client sampler due to insufficient available clients
                break
            elif channel == schedule_ch:
                round_idx, chunk_idx, phase_idx = pickle.loads(raw_data)
                # logging.info(f"[Scheduler] {round_idx}/{chunk_idx}/{phase_idx}.")
                current_phase = self.round_dict[round_idx][phase_idx]
                if current_phase.chunks_completed[chunk_idx]:
                    continue  # redundant signal
                self.round_dict[round_idx][phase_idx]\
                    .chunks_completed[chunk_idx] = True

                # if there are some phases from other chunks that
                # are waiting for its completion (for resources)
                waited_by = current_phase.waited_by
                if chunk_idx in waited_by:
                    waiter_dict = waited_by[chunk_idx]
                    for _chunk_idx, _phase_idx_list in waiter_dict.items():
                        for _phase_idx in _phase_idx_list:
                            resource_available = self.if_resource_available(
                                round_idx=round_idx,
                                chunk_idx=_chunk_idx,
                                phase_idx=_phase_idx
                            )
                            if resource_available:
                                self.hyades_server.protocol.execute_a_task(
                                    task_info=(round_idx, _chunk_idx, _phase_idx)
                                )

                if phase_idx == self.final_node_id: # if it is the final phase
                    if not (False in current_phase.chunks_completed) \
                            and round_idx == self.round:
                        logging.info("%s Round %d ended.",
                                     self.log_prefix_str, self.round)
                        if self.round < Config().app.repeat - 1:
                            self.round += 1
                            logging.info("%s Starting round %d.",
                                         self.log_prefix_str, self.round)
                            self.new_a_round(  # other condition can also lead to termination
                                round_idx=self.round,
                                num_chunks=self.num_chunks
                            )
                        else:
                            logging.info(f"Reaching the maximum number of rounds.")
                            break
                else:
                    # start its successor if possible
                    successor_phase_idx = current_phase.successor_id

                    # first conditions: the predecessors of the successor are all done
                    all_siblings_done = self.if_all_siblings_done(
                        round_idx=round_idx,
                        chunk_idx=chunk_idx,
                        phase_idx=successor_phase_idx
                    )
                    if not all_siblings_done:
                        return

                    if successor_phase_idx == self.final_node_id:
                        # second conditions: when it reaches the final chunk
                        if chunk_idx == self.num_chunks - 1:
                            for target_chunk in range(self.num_chunks):
                                self.hyades_server.protocol.execute_a_task(
                                    task_info=(round_idx, target_chunk, successor_phase_idx)
                                )
                    else:
                        # second conditions: related resource is not used by
                        # the previous chunk is done
                        resource_available = self.if_resource_available(
                            round_idx=round_idx,
                            chunk_idx=chunk_idx,
                            phase_idx=successor_phase_idx,
                            set_waited=True
                        )

                        if resource_available:
                            self.hyades_server.protocol.execute_a_task(
                                task_info=(round_idx, chunk_idx, successor_phase_idx)
                            )
