import subprocess
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from paramiko.client import SSHClient, AutoAddPolicy
from datetime import datetime


def my_random_zipfian(a, n, amin, amax, seed=None):
    prob = np.array([1 / k**a for k
                     in np.arange(1, n + 1)])
    res = [(e - min(prob)) / (max(prob) - min(prob)) * (amax - amin) + amin for e in prob]
    res = [round(e, 2) for e in res]

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(res)
    else:
        np.random.shuffle(res)
    return res


def get_chunks_idx(l, n):
    d, r = divmod(l, n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def get_chunks_idx_with_mod(l, n, mod):
    group_num = (l - 1) // mod + 1
    d, r = divmod(group_num, n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        begin = si * mod
        end = (si + (d + 1 if i < r else d)) * mod
        if end > l:
            end = l
        yield begin, end


def calc_sleep_time(sec_per_step, cur_step, start_time, gap=0):
    expected_time = sec_per_step * cur_step
    actual_time = time.perf_counter() - start_time
    start_time_drift = actual_time - expected_time
    sleep_time = max(gap, sec_per_step - start_time_drift)
    return sleep_time


def execute_locally(commands, shell=False):
    resps = []
    for command in commands:
        background = False
        if shell:
            # process = subprocess.Popen(command,
            #                            shell=True,
            #                            stdout=subprocess.PIPE)
            # only use it in shaohuai's machines as their default is sh
            # otherwise will encounter problems like "scp has no option -q"
            process = subprocess.Popen(command,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       executable="/bin/bash")
            if command.split()[-1] == "&":
                background = True
        else:
            # process = subprocess.Popen(command.split(),
            #                            shell=False,
            #                            stdout=subprocess.PIPE)
            # only use it in shaohuai's machines as their default is sh
            # otherwise will encounter problems like "scp has no option -q"
            process = subprocess.Popen(command,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       executable="/bin/bash")

        if not background:
            resp = process.communicate()[0].strip().decode("utf-8")
        else:
            resp = "None (Background job contained)."
        resps.append({
            'local_command': command,
            'time': datetime.now().strftime("(%Y-%m-%d) %H:%M:%S"),
            'resp': resp,
        })
    return resps


def ends_with_and(command):
    l = command.split(' ')
    for i in range(len(l)-1, -1, -1):
        word = l[i]
        if len(word) == 0:
            continue
        if word == '&':
            return True
        else:
            return False
    return False


def execute_remotely(commands, hostname, username, key_filename):
    while True:
        try:
            ssh_client = SSHClient()
            ssh_client.set_missing_host_key_policy(AutoAddPolicy())
            ssh_client.connect(hostname, username=username, key_filename=key_filename,
                               banner_timeout=300)
        except Exception as e:
            print(f'Encountered exception: {str(e)}, will retry soon ...')
            time.sleep(1)
        else:
            break

    resps = []
    for command in commands:
        _, stdout, stderr = ssh_client.exec_command(command)
        if not ends_with_and(command):
            resps.append({
                'remote_command': command,
                'time': datetime.now().strftime("(%Y-%m-%d) %H:%M:%S"),
                'stdout': stdout.readlines(),
                'stderr': stderr.readlines(),
            })
        else:
            resps.append({
                'remote_command': command,
                'time': datetime.now().strftime("(%Y-%m-%d) %H:%M:%S"),
                'stdout': "",
                'stderr': "",
            })
    ssh_client.close()
    return resps


def execute_for_a_node(execution_plan):
    name = execution_plan['name']
    if "public_ip" in execution_plan:
        public_ip = execution_plan['public_ip']
    execution_sequence = execution_plan['execution_sequence']
    response = []

    for action, payload in execution_sequence:
        if action == "prompt":
            for line in payload:
                print(line)
        elif action == "local":
            response.extend(execute_locally(payload))
        elif action == "local-shell":
            response.extend(execute_locally(payload, shell=True))
        elif action == "remote":
            commands = payload["commands"]
            username = payload["username"]
            key_filename = payload["key_filename"]
            response.extend(execute_remotely(commands, public_ip,
                                             username, key_filename))

    if "public_ip" in execution_plan:
        result = {
            'name': name,
            'public_ip': public_ip,
            'response': response
        }
    else:
        result = {
            'name': name,
            'response': response
        }
    return result


class ExecutionEngine(object):
    def __init__(self):
        super(ExecutionEngine, self).__init__()
        # self.n_jobs = min(cpu_count(), 16)
        self.n_jobs = cpu_count()

    def run(self, execution_plan_list, multiplier=1):
        with Pool(processes=self.n_jobs * multiplier) as pool:
            result_list = pool.starmap(execute_for_a_node, execution_plan_list)
        return result_list