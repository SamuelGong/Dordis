import logging
import numpy as np
from dordis.config import Config
from dordis.client_samplers import base
from dordis.utils.share_memory_handler \
    import SAMPLED_CLIENTS, CLIENT_STATS
import math
import random
from random import Random
from collections import OrderedDict
import logging


# To make it runnable in multiprocessing
TOTAL_ARMS = "total_arms"
UNEXPLORED = "unexplored"
SUCCESSFUL_CLIENTS = "successful_clients"
TRAINING_ROUND = "training_round"
ROUND_PREFER_DURATION = "round_prefer_duration"
LAST_UTIL_RECORD = "last_util_record"
EXPLOIT_UTIL_HISTORY = "exploitUtilHistory"
EXPLORE_UTIL_HISTORY = "exploreUtilHistory"
EXPLOIT_CLIENTS = "exploitClients"
EXPLORE_CLIENTS = "exploreClients"
BLACKLIST = "blacklist"
ROUND_THRESHOLD = "round_threshold"
EXPLORATION = "exploration"


def create_training_selector(client_id, args, sample_seed=1):
    return _training_selector(client_id, args, sample_seed)


# add ShareBase as parent class to make it runnable in multiprocessing
class _training_selector(ShareBase):
    """Oort's training selector
    """
    def __init__(self, client_id, args, sample_seed=1):
        ShareBase.__init__(self, client_id=client_id)

        # cannot make args a member, otherwise the selector cannot be pickled
        # (from the need for decoupling testing and training at server)
        self.decay_factor = args.exploration_decay
        self.exploration_min = args.exploration_min
        self.alpha = args.exploration_alpha
        self.sample_window = args.sample_window
        self.pacer_step = args.pacer_step
        self.pacer_delta = args.pacer_delta
        self.blacklist_rounds = args.blacklist_rounds
        self.blacklist_max_len = args.blacklist_max_len
        self.clip_bound = args.clip_bound
        self.round_penalty = args.round_penalty
        self.cut_off_util = args.cut_off_util

        self.rng = Random()
        self.rng.seed(sample_seed)
        np.random.seed(sample_seed)

        self.set_a_shared_value(key=EXPLORATION, value=args.exploration_factor)
        self.set_a_shared_value(key=ROUND_THRESHOLD, value=args.round_threshold)
        self.set_a_shared_value(key=TOTAL_ARMS, value=OrderedDict())
        self.set_a_shared_value(key=UNEXPLORED, value=set())
        self.set_a_shared_value(key=SUCCESSFUL_CLIENTS, value=set())
        self.set_a_shared_value(key=TRAINING_ROUND, value=0)
        self.set_a_shared_value(key=ROUND_PREFER_DURATION, value=float('inf'))
        self.set_a_shared_value(key=LAST_UTIL_RECORD, value=0)
        self.set_a_shared_value(key=EXPLOIT_UTIL_HISTORY, value=[])
        self.set_a_shared_value(key=EXPLORE_UTIL_HISTORY, value=[])
        self.set_a_shared_value(key=EXPLOIT_CLIENTS, value=[])
        self.set_a_shared_value(key=EXPLORE_CLIENTS, value=[])
        self.set_a_shared_value(key=BLACKLIST, value=None)

    def register_client(self, clientId, feedbacks):
        # Initiate the score for arms.
        # [score, time_stamp, # of trials, size of client, auxi, duration]
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        training_round = self.get_a_shared_value(key=TRAINING_ROUND)
        unexplored = self.get_a_shared_value(key=UNEXPLORED)

        totalArms.update({
            clientId: {
                'reward': feedbacks['reward'],
                'duration': feedbacks['duration'],
                'time_stamp': training_round,
                'count': 0,
                'status': True
            }
        })
        unexplored.add(clientId)

        self.set_a_shared_value(key=TOTAL_ARMS, value=totalArms)
        self.set_a_shared_value(key=UNEXPLORED, value=unexplored)

    def is_reward_updated(self, client_id):
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        return totalArms[client_id]['reward']

    def calculateSumUtil(self, clientList):
        successfulClients = self.get_a_shared_value(key=SUCCESSFUL_CLIENTS)
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)

        cnt, cntUtil = 1e-4, 0
        for client in clientList:
            if client in successfulClients:
                cnt += 1
                cntUtil += totalArms[client]['reward']
        return cntUtil / cnt

    def pacer(self):
        exploreUtilHistory = self.get_a_shared_value(key=EXPLORE_UTIL_HISTORY)
        exploitUtilHistory = self.get_a_shared_value(key=EXPLOIT_UTIL_HISTORY)
        exploreClients = self.get_a_shared_value(key=EXPLORE_CLIENTS)
        exploitClients = self.get_a_shared_value(key=EXPLOIT_CLIENTS)
        training_round = self.get_a_shared_value(key=TRAINING_ROUND)
        last_util_record = self.get_a_shared_value(key=LAST_UTIL_RECORD)
        round_threshold = self.get_a_shared_value(key=ROUND_THRESHOLD)

        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(exploreClients)
        lastExploitationUtil = self.calculateSumUtil(exploitClients)
        exploreUtilHistory.append(lastExplorationUtil)
        exploitUtilHistory.append(lastExploitationUtil)

        if training_round >= 2 * self.pacer_step \
                and training_round % self.pacer_step == 0:

            utilLastPacerRounds = sum(
                exploitUtilHistory[-2 *
                                   self.pacer_step:-self.pacer_step])
            utilCurrentPacerRounds = sum(
                exploitUtilHistory[-self.pacer_step:])

            # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
            if abs(utilCurrentPacerRounds -
                   utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                round_threshold = min(100., round_threshold + self.pacer_delta)
                last_util_record = training_round - self.pacer_step
                logging.debug(
                    "[Oort] Training selector: Pacer changes at {} to {}".format(
                        training_round, round_threshold))

            # change sharply -> we decrease the pacer step
            elif abs(utilCurrentPacerRounds -
                     utilLastPacerRounds) >= utilLastPacerRounds * 5:
                round_threshold = max(self.pacer_delta,
                    round_threshold - self.pacer_delta)
                last_util_record = training_round - self.pacer_step
                logging.debug(
                    "[Oort] Training selector: Pacer changes at {} to {}".format(
                        training_round, round_threshold))

            logging.debug(
                "[Oort] Training selector: utilLastPacerRounds {}, utilCurrentPacerRounds {} in round {}"
                .format(utilLastPacerRounds, utilCurrentPacerRounds,
                        training_round))

        self.set_a_shared_value(key=EXPLORE_UTIL_HISTORY, value=exploreUtilHistory)
        self.set_a_shared_value(key=EXPLOIT_UTIL_HISTORY, value=exploitUtilHistory)
        self.set_a_shared_value(key=LAST_UTIL_RECORD, value=last_util_record)
        self.set_a_shared_value(key=ROUND_THRESHOLD, value=round_threshold)
        logging.info(
            "[Oort] Training selector: Pacer {}: lastExploitationUtil {}, "
            "lastExplorationUtil {}, last_util_record {}"
            .format(training_round, lastExploitationUtil,
                    lastExplorationUtil, last_util_record))

    def update_client_util(self, clientId, feedbacks):
        '''
        @ feedbacks['reward']: statistical utility
        @ feedbacks['duration']: system utility
        @ feedbacks['count']: times of involved
        '''
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        unexplored = self.get_a_shared_value(key=UNEXPLORED)
        successfulClients = self.get_a_shared_value(key=SUCCESSFUL_CLIENTS)

        totalArms[clientId]['reward'] = feedbacks['reward']
        totalArms[clientId]['duration'] = feedbacks['duration']
        totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        totalArms[clientId]['count'] += 1
        totalArms[clientId]['status'] = feedbacks['status']
        unexplored.discard(clientId)
        successfulClients.add(clientId)

        self.set_a_shared_value(key=TOTAL_ARMS, value=totalArms)
        self.set_a_shared_value(key=UNEXPLORED, value=unexplored)
        self.set_a_shared_value(key=SUCCESSFUL_CLIENTS, value=successfulClients)

    def get_blacklist(self):
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)

        blacklist = []
        if self.blacklist_rounds != -1:
            sorted_client_ids = sorted(
                list(totalArms),
                reverse=True,
                key=lambda k: totalArms[k]['count'])

            for clientId in sorted_client_ids:
                if totalArms[clientId]['count'] > self.blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break

            # we need to back up if we have blacklisted all clients
            predefined_max_len = self.blacklist_max_len * len(totalArms)

            if len(blacklist) > predefined_max_len:
                logging.warning(
                    "[Oort] Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]

        return set(blacklist)

    def select_participant(self, num_of_clients, log_prefix_str,
                           feasible_clients=None, seed=None):
        '''
        @ num_of_clients: # of clients selected
        '''
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        training_round = self.get_a_shared_value(key=TRAINING_ROUND)

        if not num_of_clients:
            return []
        else:
            viable_clients = feasible_clients if feasible_clients is not None else set(
                [x for x in totalArms.keys() if totalArms[x]['status']])
            return self.getTopK(num_of_clients, training_round + 1,
                                viable_clients, log_prefix_str, seed)

    def update_duration(self, clientId, duration):
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        if clientId in totalArms:
            totalArms[clientId]['duration'] = duration
        self.set_a_shared_value(key=TOTAL_ARMS, value=totalArms)

    def getTopK(self, numOfSamples, cur_time,
                feasible_clients, log_prefix_str, seed=None):
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        round_threshold = self.get_a_shared_value(key=ROUND_THRESHOLD)
        exploration = self.get_a_shared_value(key=EXPLORATION)
        unexplored = self.get_a_shared_value(key=UNEXPLORED)

        training_round = cur_time
        blacklist = self.get_blacklist()
        self.pacer()

        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        # actual_explore_len = 0
        client_list = list(totalArms.keys())
        orderedKeys = [
            x for x in client_list
            if int(x) in feasible_clients and int(x) not in blacklist
        ]

        if round_threshold < 100.:
            sortedDuration = sorted(
                [totalArms[key]['duration'] for key in client_list])
            round_prefer_duration = sortedDuration[min(
                int(len(sortedDuration) * round_threshold / 100.),
                len(sortedDuration) - 1)]
        else:
            round_prefer_duration = float('inf')

        moving_reward, staleness, allloss = [], [], {}
        for clientId in orderedKeys:
            if totalArms[clientId]['reward'] > 0:
                creward = totalArms[clientId]['reward']
                moving_reward.append(creward)
                staleness.append(cur_time - totalArms[clientId]['time_stamp'])

        max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(
            moving_reward, self.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(
            staleness, thres=1)

        for key in orderedKeys:
            # we have played this arm before
            if totalArms[key]['count'] > 0:
                creward = min(totalArms[key]['reward'], clip_value)
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) \
                    + math.sqrt(0.1*math.log(cur_time)
                                /totalArms[key]['time_stamp']) # temporal uncertainty

                clientDuration = totalArms[key]['duration']
                if clientDuration > round_prefer_duration:
                    sc *= (
                        (float(round_prefer_duration) /
                         max(1e-4, clientDuration)) ** self.round_penalty)

                if totalArms[key]['time_stamp'] == cur_time:
                    allloss[key] = sc

                scores[key] = abs(sc)

        # Zhifeng's patch and comments
        clientLakes = list(scores.keys())  # scores: explored and feasible clients
        exploration = max(exploration * self.decay_factor, self.exploration_min)
        _unexplored = [  # unexplored and feasible
            x for x in list(unexplored) if int(x) in feasible_clients
        ]
        # why not the orginal int(numOfSamples * self.exploration):
        # to be friendly when numOfSamples is small--especially for adaption to asynchronous training
        planned_explore_len = min(len(_unexplored), np.random.binomial(
            numOfSamples, exploration, 1)[0])
        actual_exploit_len = min(numOfSamples - planned_explore_len, len(clientLakes))
        actual_explore_len = min(numOfSamples - actual_exploit_len, len(_unexplored))

        # exploitation
        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)
        # take cut-off utility
        if len(sortedClientUtil) > 0:  # Zhifeng's patch
            cut_off_index = actual_exploit_len \
                if actual_exploit_len < len(sortedClientUtil) \
                else len(sortedClientUtil) - 1
            cut_off_util = scores[
                sortedClientUtil[cut_off_index]] * self.cut_off_util

            tempPickedClients = []
            for clientId in sortedClientUtil:
                # we want at least 10 times of clients for augmentation
                if scores[clientId] < cut_off_util and len(
                        tempPickedClients) > 10. * actual_exploit_len:
                    break
                tempPickedClients.append(clientId)

            augment_factor = len(tempPickedClients)

            # Zhifeng's patch: avoid probabilities do not sum to 1
            # while preseving numerical stability
            totalSc = sum([scores[key] for key in tempPickedClients])
            if totalSc == 0:
                p = None
            else:
                p = [scores[key] / totalSc for key in tempPickedClients]

            if seed is None:
                exploitClients = list(
                    np.random.choice(
                        tempPickedClients,
                        actual_exploit_len,
                        p=p,
                        replace=False)
                        .astype(object)
                )  # Zhifeng's patch: not numpy.int
            else:  # for Lotto's verifiable sampling
                logging.info(f"{log_prefix_str} Determining exploitClients "
                             f"using seed {seed}.")
                rng = np.random.default_rng(seed=seed)
                exploitClients = list(
                    rng.choice(
                        tempPickedClients,
                        actual_exploit_len,
                        p=p,
                        replace=False)
                        .astype(object)
                )  # Zhifeng's patch: not numpy.int
        else:
            exploitClients = list()
            augment_factor = 0

        # exploration
        if actual_explore_len > 0:
            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = totalArms[cl]['reward']
                clientDuration = totalArms[cl]['duration']

                if clientDuration > round_prefer_duration:
                    init_reward[cl] *= ((float(round_prefer_duration) /
                         max(1e-4, clientDuration)) ** self.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            actual_explore_len = min(len(_unexplored),
                             numOfSamples - len(exploitClients))

            # Zhifeng's patch for load balancing
            keys = list(init_reward.keys())
            np.random.shuffle(keys)
            temp_init_reward = {k: init_reward[k] for k in keys}
            init_reward = temp_init_reward

            pickedUnexploredClients = sorted(
                init_reward, key=init_reward.get,
                reverse=True)[:min(int(self.sample_window
                                       * actual_explore_len), len(init_reward))]
            unexploredSc = float(
                sum([init_reward[key] for key in pickedUnexploredClients]))

            # Zhifeng's patch: avoid probabilities do not sum to 1
            # while preseving numerical stability
            if unexploredSc == 0:
                p = None
            else:
                p = [
                    init_reward[key] / unexploredSc
                    for key in pickedUnexploredClients
                ]

            if seed is None:
                pickedUnexplored = list(
                    np.random.choice(pickedUnexploredClients,
                                     actual_explore_len,
                                     p=p,
                                     replace=False).astype(object)
                ) # Zhifeng's patch: not numpy.int
            else:  # for Lotto's verifiable sampling
                logging.info(f"{log_prefix_str} Determining pickedUnexplored "
                             f"using seed {seed}.")
                rng = np.random.default_rng(seed=seed)
                pickedUnexplored = list(
                    rng.choice(pickedUnexploredClients,
                                     actual_explore_len,
                                     p=p,
                                     replace=False).astype(object)
                )  # Zhifeng's patch: not numpy.int

            exploreClients = pickedUnexplored
        else:
            exploreClients = list()

        pickedClients = exploreClients + exploitClients
        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            _score = (totalArms[clientId]['reward'] -
                      min_reward) / range_reward
            _staleness = self.alpha * (
                (cur_time - totalArms[clientId]['time_stamp']) -
                min_staleness
            ) / float(
                range_staleness
            )  #math.sqrt(0.1*math.log(cur_time)/max(1e-4, self.totalArms[clientId]['time_stamp']))
            top_k_score.append((totalArms[clientId], [_score, _staleness]))

        self.set_a_shared_value(key=TRAINING_ROUND, value=training_round)
        self.set_a_shared_value(key=BLACKLIST, value=blacklist)
        self.set_a_shared_value(key=ROUND_PREFER_DURATION, value=round_prefer_duration)
        self.set_a_shared_value(key=EXPLORATION, value=exploration)
        self.set_a_shared_value(key=EXPLOIT_CLIENTS, value=exploitClients)
        self.set_a_shared_value(key=EXPLORE_CLIENTS, value=exploreClients)

        logging.info(f"{log_prefix_str} [Oort] "
                     f"[Debug] planned_explore_len: {planned_explore_len}, "
                     f"actual_exploit_len {actual_exploit_len}, "
                     f"actual_explore_len {actual_explore_len}, "
                     f"explored and feasible: {clientLakes}, "
                     f"scores for them: {scores}, "
                     f"sortedClientUtil: {sortedClientUtil}, "
                     f"unexplored and feasible: {_unexplored}, "
                     f"picked clients: {pickedClients}.")

        logging.info(
            "{} [Oort] At round {}, UCB exploited {}, "
            "augment_factor {}, exploreLen {}, un-explored {}, "
            "exploration {}, round_threshold {}, sampled score is {}"
            .format(log_prefix_str, cur_time, numOfExploited,
                    augment_factor / max(1e-4, actual_exploit_len), actual_explore_len,
                    len(unexplored), exploration, round_threshold, top_k_score))
        return pickedClients

    def get_median_reward(self):
        totalArms = self.get_a_shared_value(key=TOTAL_ARMS)
        blacklist = self.get_a_shared_value(key=BLACKLIST)

        feasible_rewards = [
            totalArms[x]['reward'] for x in list(totalArms.keys())
            if int(x) not in blacklist
        ]

        # we report mean instead of median
        if len(feasible_rewards) > 0:
            return sum(feasible_rewards) / float(len(feasible_rewards))
        return 0

    def get_client_reward(self, armId):
        return self.get_a_shared_value(key=TOTAL_ARMS)[armId]

    def getAllMetrics(self):
        return self.get_a_shared_value(key=TOTAL_ARMS)

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList) * clip_bound), len(aList) - 1)]

        _max = aList[-1]
        _min = aList[0] * 0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList) / max(1e-4, float(len(aList)))

        return float(_max), float(_min), float(_range), float(_avg), float(
            clip_value)


class ClientSampler(base.ClientSampler):
    def __init__(self, client_id):
        super(ClientSampler, self).__init__(client_id)
        # seed = Config().clients.sample.seed
        # np.random.seed(seed)

        self.internal_selector = create_training_selector(
            client_id,
            Config().clients.sample.params
        )

        if hasattr(Config().clients.sample, "sample_size"):
            assert hasattr(Config().clients, "worst_online_frac")
            self.sample_size = Config().clients.sample.sample_size
            self.worst_online_frac = Config().clients.worst_online_frac
            self.num_sampled_client_upperbound = self.sample_size
            self.num_worst_online_clients \
                = int(np.floor(self.total_clients * self.worst_online_frac))
            self.sampling_rate_upperbound = self.sample_size \
                                            / self.num_worst_online_clients
            self.mode = "fixed_sample_size"
        else:
            assert hasattr(Config().clients.sample,
                           "sampling_rate_upperbound")
            self.sampling_rate_upperbound \
                = Config().clients.sample.sampling_rate_upperbound
            self.num_sampled_client_upperbound = int(np.floor(
                self.total_clients * self.sampling_rate_upperbound
            ))
            self.mode = "sampling_rate_upperbounded"

        logging.info(f"To sample a subset of "
                     f"all available clients without replacement"
                     f"according to Oort's principles "
                     f"where the sampling rate for each client "
                     f"does not exceed {self.sampling_rate_upperbound}.")

    def register_a_client(self, client_id, feedback):
        self.internal_selector.register_client(
            clientId=client_id,
            feedbacks=feedback
        )
        # logging.info(f"[Oort] [Debug] Client {client_id} "
        #              f"registered with meta: {feedback}.")

    def update_a_client(self, client_id, feedback):
        self.internal_selector.update_client_util(
            clientId=client_id,
            feedbacks=feedback
        )
        # logging.info(f"[Oort] [Debug] Client {client_id} "
        #              f"updated with meta: {feedback}.")

    def pull_status_quo(self, clients, log_prefix_str):
        registered_clients = []
        updated_clients = []

        for client_id in clients:
            client_dict = self.get_a_shared_value(
                key=[CLIENT_STATS, client_id]
            )
            for round_idx, round_dict in client_dict["round_stats"].items():
                if round_dict["used"] is False:
                    if round_idx == -1:  # for registration
                        self.register_a_client(
                            client_id=client_id,
                            feedback={
                                'reward': client_dict["training_dataset_size"],
                                'duration': round_dict["time"]
                            }
                        )
                        registered_clients.append(client_id)
                    else:  # for update
                        reward = client_dict["training_dataset_size"] \
                                 * round_dict["utility"]
                        self.update_a_client(
                            client_id=client_id,
                            feedback={
                                'reward': reward,
                                'duration': round_dict["time"],
                                # cannot start from 0
                                'time_stamp': round_idx + 1,
                                'status': True,
                            }
                        )
                        updated_clients.append(client_id)

                    round_dict["used"] = True

            # updating the "used" status
            self.set_a_shared_value(
                key=[CLIENT_STATS, client_id],
                value=client_dict
            )

        logging.info(f"{log_prefix_str} [Oort] Status pulled. "
                     f"Clients registered: {registered_clients}, "
                     f"clients updated: {updated_clients}.")

        # for debug use
        all_dict = self.internal_selector.getAllMetrics()
        clean_dict = {
            k: round(all_dict[k]['reward'], 2) for k in all_dict.keys()
        }
        logging.info(f"{log_prefix_str} [Oort] [Debug] {clean_dict}.")

    def get_num_sampled_clients_upperbound(self):
        return self.num_sampled_client_upperbound

    def get_sampling_rate_upperbound(self):
        return 1.0  # this is for DP's use: we do not have any
        # randomness for the server
        # return self.sampling_rate_upperbound

    def sample(self, candidates, round_idx, log_prefix_str,
               seed=None, save_result=True):
        # first pull new state
        self.pull_status_quo(candidates, log_prefix_str)

        # then select
        if self.mode == "fixed_sample_size":
            sample_size = self.sample_size
        else:  # self.mode == "sampling_rate_upperbounded"
            sample_size \
                = int(np.floor(self.sampling_rate_upperbound * len(candidates)))

        if len(candidates) >= sample_size and len(candidates) > 0:
            sampled_clients = self.internal_selector.select_participant(
                num_of_clients=sample_size,
                log_prefix_str=log_prefix_str,
                feasible_clients=candidates,
                seed=seed
            )

            # for serialization in multiprocessing
            sampled_clients = sorted(sampled_clients)
            if save_result:
                self.set_a_shared_value(
                    key=[SAMPLED_CLIENTS, round_idx],
                    value=sampled_clients
                )
            else:
                return sampled_clients
        else:  # has to stop due to privacy and any other practical concerns
            logging.info(f"[Oort] No clients are sampled "
                         f"due to insufficient candidates "
                         f"({len(candidates)}<{sample_size}).")
